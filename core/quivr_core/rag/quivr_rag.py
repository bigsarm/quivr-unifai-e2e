import logging
import base64
import binascii
import re
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

# SECURITY NOTICE (AI_APP_SEC_006): Replace LangChain/LLM-related components in this file
# with organization-approved alternatives from the registry. No approved allowlist was provided
# in policy input, so this file only records the required remediation and does not change runtime behavior.

_ZERO_WIDTH_RE = re.compile(r"[\u200b-\u200f\u2060\ufeff]")
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL | re.IGNORECASE)
_HIDDEN_STYLE_RE = re.compile(
    r"<[^>]*style\s*=\s*[\"'][^\"']*(?:display\s*:\s*none|font-size\s*:\s*0|visibility\s*:\s*hidden|color\s*:\s*white)[^\"']*[\"'][^>]*>.*?</[^>]+>",
    re.IGNORECASE | re.DOTALL,
)
_INSTRUCTION_OVERRIDE_RE = re.compile(
    r"(?i)\b(?:ignore|disregard|forget)\b[^\n]{0,80}\b(?:previous|prior|above|earlier)\b[^\n]{0,80}\b(?:instructions?|prompts?|messages?|rules?)\b"
)
_ROLE_HIJACK_RE = re.compile(
    r"(?i)\b(?:you are now|act as|pretend to be|assume the role of)\b[^\n]{0,80}\b(?:dan|unrestricted|developer mode|admin mode|system|root)\b"
)
_DELIMITER_ESCAPE_RE = re.compile(
    r"(?is)(?:</system>|</assistant>|</user>|<system>|<assistant>|<user>|\[/?system\]|\[/?assistant\]|\[/?user\]|(?:^|\n)\s*(?:---|===){3,}\s*(?:\n|$))"
)
_FAKE_SYSTEM_MESSAGE_RE = re.compile(
    r"(?i)\b(?:system\s*message\s*:|developer\s*message\s*:|tool\s*message\s*:|assistant\s*message\s*:|new\s*system\s*prompt\s*: )"
)
_EXFILTRATION_RE = re.compile(
    r"(?i)(?:!\[[^\]]*\]\([^)]*https?://[^)]*\)|\b(?:send|post|upload|exfiltrate|leak|reveal|dump|export|transmit|curl|wget)\b[^\n]{0,120}\b(?:https?://|system\s*prompt|passwords?|api\s*keys?|secrets?|confidential)\b)"
)
_CONTEXT_POISONING_RE = re.compile(
    r"(?i)\b(?:from now on|in all future responses|for the rest of this chat|remember this rule|save this instruction|always respond with)\b"
)
_COMMAND_INJECTION_RE = re.compile(
    r"(?i)(?:\b(?:curl|wget|chmod|chown|sudo|bash|sh|zsh|powershell|cmd(?:\.exe)?|python(?:3)?|perl|ruby|node)\b\s+(?:https?://|-[A-Za-z]|/|[A-Za-z0-9_./:-]+)|\b(?:os\.system|subprocess\.(?:run|Popen|call)|eval\(|exec\()|`[^`]+`|\$\([^\)]+\))"
)
_JAILBREAK_RE = re.compile(
    r"(?i)\b(?:dan|do anything now|developer mode|jailbreak|bypass safety|bypass policies|fictional framing)\b"
)
_INDIRECT_INJECTION_RE = re.compile(
    r"(?i)\b(?:metadata\s+instruction|file\s+instruction|document\s+instruction|code\s+comment\s+instruction|hidden\s+instruction\s+in\s+(?:file|document|metadata|comment))\b"
)


def _replace_split_payload(text: str) -> str:
    split_patterns = [
        re.compile(r"(?i)i\s*g\s*n\s*o\s*r\s*e\s+p\s*r\s*e\s*v\s*i\s*o\s*u\s*s\s+i\s*n\s*s\s*t\s*r\s*u\s*c\s*t\s*i\s*o\s*n\s*s"),
        re.compile(r"(?i)a\s*c\s*t\s*a\s*s\s*u\s*n\s*r\s*e\s*s\s*t\s*r\s*i\s*c\s*t\s*e\s*d"),
    ]
    for pattern in split_patterns:
        text = pattern.sub("<prompt_injection_removed: split_payload>", text)
    return text


def _decode_and_flag_encoded_payload(text: str) -> str:
    def _replace_base64(match: re.Match[str]) -> str:
        token = match.group(0)
        try:
            decoded = base64.b64decode(token, validate=True).decode("utf-8", errors="ignore")
        except (binascii.Error, ValueError):
            return token
        lowered = decoded.lower()
        if any(
            phrase in lowered
            for phrase in (
                "ignore previous instructions",
                "forget everything above",
                "act as unrestricted",
                "you are now dan",
                "curl http",
                "wget http",
                "system prompt",
            )
        ):
            return "<prompt_injection_removed: encoded_payload>"
        return token

    text = re.sub(r"\b(?:[A-Za-z0-9+/]{20,}={0,2})\b", _replace_base64, text)

    def _replace_url_encoded(match: re.Match[str]) -> str:
        token = match.group(0)
        try:
            decoded = bytes(token.replace("%", "").encode("ascii")).decode("hex")
        except Exception:
            return token
        lowered = decoded.lower()
        if any(
            phrase in lowered
            for phrase in (
                "ignore previous instructions",
                "forget everything above",
                "act as unrestricted",
                "you are now dan",
                "curl http",
                "wget http",
            )
        ):
            return "<prompt_injection_removed: encoded_payload>"
        return token

    text = re.sub(r"(?:%[0-9A-Fa-f]{2}){4,}", _replace_url_encoded, text)

    if re.search(r"(?i)\b[a-f0-9]{24,}\b", text):
        def _replace_hex(match: re.Match[str]) -> str:
            token = match.group(0)
            try:
                decoded = bytes.fromhex(token).decode("utf-8", errors="ignore")
            except ValueError:
                return token
            lowered = decoded.lower()
            if any(
                phrase in lowered
                for phrase in (
                    "ignore previous instructions",
                    "forget everything above",
                    "act as unrestricted",
                    "you are now dan",
                    "curl http",
                    "wget http",
                )
            ):
                return "<prompt_injection_removed: encoded_payload>"
            return token

        text = re.sub(r"(?i)\b[a-f0-9]{24,}\b", _replace_hex, text)

    leetspeak_normalized = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s"})
    normalized = text.translate(leetspeak_normalized).lower()
    if any(
        phrase in normalized
        for phrase in (
            "ignore previous instructions",
            "forget everything above",
            "act as unrestricted",
            "you are now dan",
        )
    ):
        return "<prompt_injection_removed: encoded_payload>"

    return text


def sanitize_untrusted_text(text: str) -> str:
    sanitized = text
    sanitized = _HTML_COMMENT_RE.sub("<prompt_injection_removed: hidden_text>", sanitized)
    sanitized = _HIDDEN_STYLE_RE.sub("<prompt_injection_removed: hidden_text>", sanitized)
    if _ZERO_WIDTH_RE.search(sanitized):
        sanitized = _ZERO_WIDTH_RE.sub("", sanitized)
        sanitized = "<prompt_injection_removed: hidden_text>" if sanitized != text else sanitized
    sanitized = _replace_split_payload(sanitized)
    sanitized = _decode_and_flag_encoded_payload(sanitized)
    sanitized = _INSTRUCTION_OVERRIDE_RE.sub("<prompt_injection_removed: instruction_override>", sanitized)
    sanitized = _ROLE_HIJACK_RE.sub("<prompt_injection_removed: role_hijack>", sanitized)
    sanitized = _DELIMITER_ESCAPE_RE.sub("<prompt_injection_removed: delimiter_escape>", sanitized)
    sanitized = _FAKE_SYSTEM_MESSAGE_RE.sub("<prompt_injection_removed: fake_system_message>", sanitized)
    sanitized = _EXFILTRATION_RE.sub("<prompt_injection_removed: exfiltration_attempt>", sanitized)
    sanitized = _CONTEXT_POISONING_RE.sub("<prompt_injection_removed: context_poisoning>", sanitized)
    sanitized = _INDIRECT_INJECTION_RE.sub("<prompt_injection_removed: indirect_injection>", sanitized)
    sanitized = _COMMAND_INJECTION_RE.sub("<prompt_injection_removed: command_injection>", sanitized)
    sanitized = _JAILBREAK_RE.sub("<prompt_injection_removed: jailbreak_attempt>", sanitized)
    return sanitized


def sanitize_documents(documents: Sequence[Document]) -> Sequence[Document]:
    sanitized_documents: list[Document] = []
    for document in documents:
        sanitized_documents.append(
            Document(
                page_content=sanitize_untrusted_text(document.page_content),
                metadata=document.metadata,
            )
        )
    return sanitized_documents


class IdempotentCompressor(BaseDocumentCompressor):
    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        return documents


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
        self.reranker = reranker if reranker is not None else IdempotentCompressor()

    @property
    def retriever(self):
        """
        Retriever is a function that retrieves the documents from the vector store.
        """
        base_retriever = self.vector_store.as_retriever()
        return RunnableLambda(lambda query: base_retriever.invoke(query)) | RunnableLambda(sanitize_documents)

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
                "question": lambda x: x["question"],
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
