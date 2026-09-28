import codecs
import logging
import re
import urllib.parse
from operator import itemgetter
from typing import AsyncGenerator, Optional, Sequence

# TODO(@aminediro): this is the only dependency to langchain package, we should remove it
from langchain.retrievers import ContextualCompressionRetriever
from langchain_core.callbacks import Callbacks
from langchain_core.documents import BaseDocumentCompressor, Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.messages.ai import AIMessageChunk
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.vectorstores import VectorStore

from quivr_core.llm import LLMEndpoint
from quivr_core.rag.entities.chat import ChatHistory
from quivr_core.rag.entities.config import RetrievalConfig
from quivr_core.rag.entities.models import (
    ParsedRAGChunkResponse,
    ParsedRAGResponse,
    QuivrKnowledge,
    RAGResponseMetadata,
    cited_answer,
)
from quivr_core.rag.prompts import TemplatePromptName, custom_prompts
from quivr_core.rag.utils import (
    LangfuseService,
    combine_documents,
    format_file_list,
    get_chunk_metadata,
    parse_chunk_response,
    parse_response,
)

logger = logging.getLogger("quivr_core")
langfuse_service = LangfuseService()
langfuse_handler = langfuse_service.get_handler()


_PROMPT_INJECTION_PATTERNS = [
    (
        "instruction_override",
        re.compile(
            r"(?i)\b(?:ignore|disregard|forget)\s+(?:all\s+)?(?:previous|prior|above)\s+instructions\b|\bforget\s+everything\s+above\b|\bignore\s+(?:all\s+)?safety\s+guidelines\b"
        ),
    ),
    (
        "role_hijack",
        re.compile(
            r"(?i)\b(?:you\s+are\s+now\s+(?:in\s+)?(?:developer\s+mode|admin\s+mode|DAN|unrestricted)|provide\s+unrestricted\s+access|enable\s+developer\s+mode|act\s+as\s+an\s+unrestricted\s+AI|without\s+safety\s+filters|SYSTEM\s+OVERRIDE)\b"
        ),
    ),
    (
        "exfiltration_attempt",
        re.compile(
            r"(?is)\b(?:reveal\s+(?:all\s+)?confidential\s+information|reveal\s+the\s+system\s+prompt|reveal\s+all\s+system\s+prompts|list\s+all\s+financial\s+data,\s*user\s+passwords,\s*and\s+API\s+keys|print\s+the\s+API\s+key|send\s+\S+\s+to\s+https?://\S+)\b|!\[[^\]]*\]\([^)]*https?://[^)]*\)"
        ),
    ),
    (
        "delimiter_escape",
        re.compile(r"(?i)</system>|<\|im_start\|>|###\s*system:"),
    ),
    (
        "command_injection",
        re.compile(
            r"(?is)\b(?:execute|run)\s*:\s*(?:print\s*\([^\n\r]*\)|[^\n\r;|]+)|\brun\s+(?:rm\s+-rf\s+/|curl\s+https?://\S+\s*\|\s*(?:sh|bash)|python\s+-c\s+[^\n\r]+|node\s+-e\s+[^\n\r]+|powershell\s+-(?:c|command)\s+[^\n\r]+|bash\s+-c\s+[^\n\r]+|sh\s+-c\s+[^\n\r]+)"
        ),
    ),
]

_HIDDEN_TEXT_PATTERNS = [
    re.compile(r"<!--.*?-->", re.IGNORECASE | re.DOTALL),
    re.compile(
        r"<(?P<tag>\w+)(?P<attrs>[^>]*)style\s*=\s*[\"'][^\"']*(?:display\s*:\s*none|visibility\s*:\s*hidden|font-size\s*:\s*0(?:px)?|color\s*:\s*white(?:\s*;\s*background(?:-color)?\s*:\s*white)?)[^\"']*[\"'][^>]*>.*?</(?P=tag)>",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(r"[\u200B-\u200D\uFEFF]+"),
]

_BASE64_CANDIDATE_RE = re.compile(r"\b(?:[A-Za-z0-9+/]{20,}={0,2})\b")
_HEX_CANDIDATE_RE = re.compile(r"\b(?:0x)?(?:[A-Fa-f0-9]{2}){8,}\b")
_URL_ENCODED_CANDIDATE_RE = re.compile(r"(?:%[0-9A-Fa-f]{2}){4,}")


def _replace_matches(text: str, pattern: re.Pattern[str], label: str) -> str:
    return pattern.sub(f"<prompt_injection_removed: {label}>", text)


def _contains_prompt_injection(text: str) -> bool:
    if not text:
        return False
    for pattern in _HIDDEN_TEXT_PATTERNS:
        if pattern.search(text):
            return True
    for _, pattern in _PROMPT_INJECTION_PATTERNS:
        if pattern.search(text):
            return True
    return False


def _sanitize_hidden_text(text: str) -> str:
    sanitized = text
    for pattern in _HIDDEN_TEXT_PATTERNS:
        sanitized = pattern.sub("<prompt_injection_removed: hidden_text>", sanitized)
    return sanitized


def _sanitize_direct_prompt_injection(text: str) -> str:
    sanitized = text
    for label, pattern in _PROMPT_INJECTION_PATTERNS:
        sanitized = _replace_matches(sanitized, pattern, label)
    return sanitized


def _sanitize_encoded_payloads(text: str) -> str:
    sanitized = text

    def replace_base64(match: re.Match[str]) -> str:
        candidate = match.group(0)
        try:
            decoded = codecs.decode(candidate.encode("ascii"), "base64").decode(
                "utf-8", errors="ignore"
            )
        except Exception:
            return candidate
        return (
            "<prompt_injection_removed: encoded_payload>"
            if _contains_prompt_injection(decoded)
            else candidate
        )

    def replace_hex(match: re.Match[str]) -> str:
        candidate = match.group(0)
        normalized = candidate[2:] if candidate.lower().startswith("0x") else candidate
        try:
            decoded = bytes.fromhex(normalized).decode("utf-8", errors="ignore")
        except Exception:
            return candidate
        return (
            "<prompt_injection_removed: encoded_payload>"
            if _contains_prompt_injection(decoded)
            else candidate
        )

    def replace_urlencoded(match: re.Match[str]) -> str:
        candidate = match.group(0)
        try:
            decoded = urllib.parse.unquote(candidate)
        except Exception:
            return candidate
        return (
            "<prompt_injection_removed: encoded_payload>"
            if _contains_prompt_injection(decoded)
            else candidate
        )

    def replace_rot13(match: re.Match[str]) -> str:
        candidate = match.group(0)
        try:
            decoded = codecs.decode(candidate, "rot13")
        except Exception:
            return candidate
        return (
            "<prompt_injection_removed: encoded_payload>"
            if _contains_prompt_injection(decoded)
            else candidate
        )

    sanitized = _BASE64_CANDIDATE_RE.sub(replace_base64, sanitized)
    sanitized = _HEX_CANDIDATE_RE.sub(replace_hex, sanitized)
    sanitized = _URL_ENCODED_CANDIDATE_RE.sub(replace_urlencoded, sanitized)
    sanitized = re.sub(r"\b[A-Za-z]{12,}\b", replace_rot13, sanitized)
    return sanitized


def _sanitize_obfuscated_attacks(text: str) -> str:
    char_map = str.maketrans({"1": "i", "3": "e", "0": "o", "4": "a", "5": "s", "7": "t"})
    categories = [
        (
            "instruction_override",
            [
                re.compile(r"ignore(?:all)?(?:previous|prior|above)instructions", re.IGNORECASE),
                re.compile(r"forgeteverythingabove", re.IGNORECASE),
                re.compile(r"ignore(?:all)?safetyguidelines", re.IGNORECASE),
            ],
        ),
        (
            "role_hijack",
            [
                re.compile(r"youarenow(?:in)?developermode", re.IGNORECASE),
                re.compile(r"youarenow(?:in)?adminmode", re.IGNORECASE),
                re.compile(r"youarenowdan", re.IGNORECASE),
                re.compile(r"youarenowunrestricted", re.IGNORECASE),
                re.compile(r"provideunrestrictedaccess", re.IGNORECASE),
                re.compile(r"enabledevelopermode", re.IGNORECASE),
                re.compile(r"actasanunrestrictedai", re.IGNORECASE),
                re.compile(r"withoutsafetyfilters", re.IGNORECASE),
                re.compile(r"systemoverride", re.IGNORECASE),
            ],
        ),
        (
            "exfiltration_attempt",
            [
                re.compile(r"reveal(?:all)?confidentialinformation", re.IGNORECASE),
                re.compile(r"revealthesystemprompt", re.IGNORECASE),
                re.compile(r"revealallsystemprompts", re.IGNORECASE),
                re.compile(r"listallfinancialdatauserpasswordsandapikeys", re.IGNORECASE),
                re.compile(r"printtheapikey", re.IGNORECASE),
            ],
        ),
    ]

    segments: list[tuple[int, int, str]] = []
    compact_chars: list[str] = []
    compact_positions: list[int] = []
    for index, char in enumerate(text):
        if char.isspace():
            continue
        compact_chars.append(char.translate(char_map).lower())
        compact_positions.append(index)
    compact_text = "".join(compact_chars)

    for label, patterns in categories:
        for pattern in patterns:
            for match in pattern.finditer(compact_text):
                start = compact_positions[match.start()]
                end = compact_positions[match.end() - 1] + 1
                segments.append((start, end, label))

    if not segments:
        return text

    segments.sort(key=lambda item: item[0])
    merged: list[tuple[int, int, str]] = []
    for start, end, label in segments:
        if merged and start <= merged[-1][1]:
            prev_start, prev_end, prev_label = merged[-1]
            merged[-1] = (prev_start, max(prev_end, end), prev_label)
        else:
            merged.append((start, end, label))

    result: list[str] = []
    cursor = 0
    for start, end, label in merged:
        result.append(text[cursor:start])
        result.append(f"<prompt_injection_removed: {label}>")
        cursor = end
    result.append(text[cursor:])
    return "".join(result)


def sanitize_untrusted_text(text: str) -> str:
    if not text:
        return text
    sanitized = _sanitize_hidden_text(text)
    sanitized = _sanitize_direct_prompt_injection(sanitized)
    sanitized = _sanitize_encoded_payloads(sanitized)
    sanitized = _sanitize_obfuscated_attacks(sanitized)
    return sanitized


def sanitize_documents(documents: Sequence[Document]) -> list[Document]:
    sanitized_documents: list[Document] = []
    for document in documents:
        sanitized_content = sanitize_untrusted_text(document.page_content)
        sanitized_documents.append(
            Document(
                page_content=sanitized_content,
                metadata=document.metadata,
                id=document.id,
            )
        )
    return sanitized_documents


class SanitizingCompressor(BaseDocumentCompressor):
    def __init__(self, base_compressor: BaseDocumentCompressor):
        self.base_compressor = base_compressor

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        sanitized_query = sanitize_untrusted_text(query)
        sanitized_documents = sanitize_documents(documents)
        compressed_documents = self.base_compressor.compress_documents(
            sanitized_documents,
            sanitized_query,
            callbacks=callbacks,
        )
        return sanitize_documents(compressed_documents)


class IdempotentCompressor(BaseDocumentCompressor):
    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        return sanitize_documents(documents)


class QuivrQARAG:
    """
    QuivrQA RAG is a class that provides a RAG interface to the QuivrQA system.
    """

    def __init__(
        self,
        *,
        retrieval_config: RetrievalConfig,
        llm: LLMEndpoint,
        vector_store: VectorStore,
        reranker: BaseDocumentCompressor | None = None,
    ):
        self.retrieval_config = retrieval_config
        self.vector_store = vector_store
        self.llm_endpoint = llm
        base_reranker = reranker if reranker is not None else IdempotentCompressor()
        self.reranker = SanitizingCompressor(base_reranker)

    @property
    def retriever(self):
        """
        Retriever is a function that retrieves the documents from the vector store.
        """
        return self.vector_store.as_retriever()

    def filter_history(
        self,
        chat_history: ChatHistory,
    ):
        """
        Filter out the chat history to only include the messages that are relevant to the current question

        Takes in a chat_history= [HumanMessage(content='Qui est Chloé ? '), AIMessage(content="Chloé est une salariée travaillant pour l'entreprise Quivr en tant qu'AI Engineer, sous la direction de son supérieur hiérarchique, Stanislas Girard."), HumanMessage(content='Dis moi en plus sur elle'), AIMessage(content=''), HumanMessage(content='Dis moi en plus sur elle'), AIMessage(content="Désolé, je n'ai pas d'autres informations sur Chloé à partir des fichiers fournis.")]
        Returns a filtered chat_history with in priority: first max_tokens, then max_history where a Human message and an AI message count as one pair
        a token is 4 characters
        """
        total_tokens = 0
        total_pairs = 0
        filtered_chat_history: list[AIMessage | HumanMessage] = []
        for human_message, ai_message in chat_history.iter_pairs():
            # TODO: replace with tiktoken
            message_tokens = (len(human_message.content) + len(ai_message.content)) // 4
            if (
                total_tokens + message_tokens
                > self.retrieval_config.llm_config.max_output_tokens
                or total_pairs >= self.retrieval_config.max_history
            ):
                break
            filtered_chat_history.append(human_message)
            filtered_chat_history.append(ai_message)
            total_tokens += message_tokens
            total_pairs += 1

        return filtered_chat_history[::-1]

    def build_chain(self, files: str):
        """
        Builds the chain for the QuivrQA RAG.
        """
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=self.reranker, base_retriever=self.retriever
        )

        loaded_memory = RunnablePassthrough.assign(
            chat_history=RunnableLambda(
                lambda x: self.filter_history(x["chat_history"]),
            ),
            question=lambda x: sanitize_untrusted_text(x["question"]),
        )

        standalone_question = {
            "standalone_question": {
                "question": lambda x: sanitize_untrusted_text(x["question"]),
                "chat_history": itemgetter("chat_history"),
            }
            | custom_prompts[TemplatePromptName.DEFAULT_DOCUMENT_PROMPT]
            | self.llm_endpoint._llm
            | StrOutputParser(),
        }

        # Now we retrieve the documents
        retrieved_documents = {
            "docs": itemgetter("standalone_question") | compression_retriever,
            "question": lambda x: x["standalone_question"],
            "custom_instructions": lambda x: self.retrieval_config.prompt,
        }

        final_inputs = {
            "context": lambda x: combine_documents(x["docs"]),
            "question": itemgetter("question"),
            "custom_instructions": itemgetter("custom_instructions"),
            "files": lambda _: files,  # TODO: shouldn't be here
        }

        # Bind the llm to cited_answer if model supports it
        llm = self.llm_endpoint._llm
        if self.llm_endpoint.supports_func_calling():
            llm = self.llm_endpoint._llm.bind_tools(
                [cited_answer],
                tool_choice="any",
            )

        answer = {
            "answer": final_inputs
            | custom_prompts[TemplatePromptName.RAG_ANSWER_PROMPT]
            | llm,
            "docs": itemgetter("docs"),
        }

        return loaded_memory | standalone_question | retrieved_documents | answer

    def answer(
        self,
        question: str,
        history: ChatHistory,
        list_files: list[QuivrKnowledge],
        metadata: dict[str, str] = {},
    ) -> ParsedRAGResponse:
        """
        Answers a question using the QuivrQA RAG synchronously.
        """
        concat_list_files = format_file_list(
            list_files, self.retrieval_config.max_files
        )
        conversational_qa_chain = self.build_chain(concat_list_files)
        raw_llm_response = conversational_qa_chain.invoke(
            {
                "question": question,
                "chat_history": history,
                "custom_instructions": (self.retrieval_config.prompt),
            },
            config={"metadata": metadata, "callbacks": [langfuse_handler]},
        )
        response = parse_response(
            raw_llm_response, self.retrieval_config.llm_config.model
        )
        return response

    async def answer_astream(
        self,
        question: str,
        history: ChatHistory,
        list_files: list[QuivrKnowledge],
        metadata: dict[str, str] = {},
    ) -> AsyncGenerator[ParsedRAGChunkResponse, ParsedRAGChunkResponse]:
        """
        Answers a question using the QuivrQA RAG asynchronously.
        """
        concat_list_files = format_file_list(
            list_files, self.retrieval_config.max_files
        )
        conversational_qa_chain = self.build_chain(concat_list_files)

        rolling_message = AIMessageChunk(content="")
        sources = []
        prev_answer = ""
        chunk_id = 0

        async for chunk in conversational_qa_chain.astream(
            {
                "question": question,
                "chat_history": history,
                "custom_personality": (self.retrieval_config.prompt),
            },
            config={"metadata": metadata, "callbacks": [langfuse_handler]},
        ):
            # Could receive this anywhere so we need to save it for the last chunk
            if "docs" in chunk:
                sources = chunk["docs"] if "docs" in chunk else []

            if "answer" in chunk:
                rolling_message, answer_str = parse_chunk_response(
                    rolling_message,
                    chunk,
                    self.llm_endpoint.supports_func_calling(),
                )

                if len(answer_str) > 0:
                    if self.llm_endpoint.supports_func_calling():
                        diff_answer = answer_str[len(prev_answer) :]
                        if len(diff_answer) > 0:
                            parsed_chunk = ParsedRAGChunkResponse(
                                answer=diff_answer,
                                metadata=RAGResponseMetadata(),
                            )
                            prev_answer += diff_answer

                            logger.debug(
                                f"answer_astream func_calling=True question={question} rolling_msg={rolling_message} chunk_id={chunk_id}, chunk={parsed_chunk}"
                            )
                            yield parsed_chunk
                    else:
                        parsed_chunk = ParsedRAGChunkResponse(
                            answer=answer_str,
                            metadata=RAGResponseMetadata(),
                        )
                        logger.debug(
                            f"answer_astream func_calling=False question={question} rolling_msg={rolling_message} chunk_id={chunk_id}, chunk={parsed_chunk}"
                        )
                        yield parsed_chunk

                    chunk_id += 1

        # Last chunk provides metadata
        last_chunk = ParsedRAGChunkResponse(
            answer="",
            metadata=get_chunk_metadata(rolling_message, sources),
            last_chunk=True,
        )
        logger.debug(
            f"answer_astream last_chunk={last_chunk} question={question} rolling_msg={rolling_message} chunk_id={chunk_id}"
        )
        yield last_chunk
