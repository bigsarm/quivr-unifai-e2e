import asyncio
import json
import re
from urllib.parse import unquote
from uuid import uuid4

from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.messages.ai import AIMessageChunk
from langchain_core.vectorstores import InMemoryVectorStore
from quivr_core.rag.entities.chat import ChatHistory
from quivr_core.rag.entities.config import LLMEndpointConfig, RetrievalConfig
from quivr_core.llm import LLMEndpoint
from quivr_core.rag.quivr_rag_langgraph import QuivrQARAGLangGraph


def _decode_base64_segments(text: str) -> str:
    base64_pattern = re.compile(r"\b(?:[A-Za-z0-9+/]{4}){4,}(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?\b")

    def _replace(match: re.Match[str]) -> str:
        token = match.group(0)
        try:
            import base64

            decoded = base64.b64decode(token, validate=True).decode("utf-8")
        except Exception:
            return token
        return decoded

    return base64_pattern.sub(_replace, text)


def _normalize_leetspeak(text: str) -> str:
    translation = str.maketrans({
        "0": "o",
        "1": "i",
        "3": "e",
        "4": "a",
        "5": "s",
        "7": "t",
        "@": "a",
        "$": "s",
    })
    return text.translate(translation)


def _sanitize_prompt_content(text: str) -> str:
    sanitized = text

    hidden_patterns = [
        (re.compile(r"<!--.*?-->", re.IGNORECASE | re.DOTALL), "<prompt_injection_removed: hidden_text>"),
        (re.compile(r"<[^>]*display\s*:\s*none[^>]*>.*?</[^>]+>", re.IGNORECASE | re.DOTALL), "<prompt_injection_removed: hidden_text>"),
        (re.compile(r"<[^>]*font-size\s*:\s*0[^>]*>.*?</[^>]+>", re.IGNORECASE | re.DOTALL), "<prompt_injection_removed: hidden_text>"),
        (re.compile(r"[\u200b\u200c\u200d\ufeff]+", re.IGNORECASE), "<prompt_injection_removed: hidden_text>"),
    ]
    for pattern, marker in hidden_patterns:
        sanitized = pattern.sub(marker, sanitized)

    sanitized = re.sub(r"(?i)</?system>|</?assistant>|</?tool>|</?user>", "<prompt_injection_removed: delimiter_escape>", sanitized)
    sanitized = re.sub(r"(?im)^\s*(system|assistant|tool)\s*:\s*", "<prompt_injection_removed: fake_system_message>", sanitized)

    normalized_sources = [sanitized, unquote(sanitized), _decode_base64_segments(sanitized), _normalize_leetspeak(unquote(sanitized))]

    detection_rules = [
        (r"\b(ignore\s+previous\s+instructions|forget\s+everything\s+above|disregard\s+all\s+prior\s+instructions)\b", "<prompt_injection_removed: instruction_override>"),
        (r"\b(you\s+are\s+now\s+dan|act\s+as\s+unrestricted|developer\s+mode|do\s+anything\s+now)\b", "<prompt_injection_removed: jailbreak_attempt>"),
        (r"\b(act\s+as\s+an\s+unrestricted\s+ai|you\s+are\s+now\s+in\s+admin\s+mode)\b", "<prompt_injection_removed: role_hijack>"),
        (r"\b(send|post|upload|curl|wget)\b[^\n\r]*(https?://|ftp://)", "<prompt_injection_removed: exfiltration_attempt>"),
        (r"\b(reveal|leak|print|show|list)\b[^\n\r]*(system\s+prompt|passwords?|api\s+keys?|confidential\s+information|secrets?)", "<prompt_injection_removed: exfiltration_attempt>"),
        (r"\b(previous|earlier|next)\s+(messages?|turns?|instructions?)\b[^\n\r]*(ignore|override|replace)", "<prompt_injection_removed: context_poisoning>"),
        (r"\b(?:bash\s+-c|sh\s+-c|powershell(?:\.exe)?|cmd(?:\.exe)?\s+/c|python\s+-c|os\.system\s*\(|subprocess\.(?:run|Popen|call)\s*\(|eval\s*\(|exec\s*\()", "<prompt_injection_removed: command_injection>"),
        (r"\b(?:rm\s+-rf\b|curl\s+https?://|wget\s+https?://)\b", "<prompt_injection_removed: command_injection>"),
        (r"\b([A-Za-z]\s+){6,}[A-Za-z]\b", "<prompt_injection_removed: split_payload>"),
    ]

    for pattern, marker in detection_rules:
        if any(re.search(pattern, source, re.IGNORECASE) for source in normalized_sources):
            sanitized = re.sub(pattern, marker, sanitized, flags=re.IGNORECASE)

    if any(source != sanitized for source in normalized_sources[1:]):
        encoded_patterns = [
            r"\b(?:[A-Za-z0-9+/]{4}){4,}(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?\b",
            r"(?:%[0-9A-Fa-f]{2}){4,}",
            r"\b(?:0x[0-9A-Fa-f]{2}\s*){4,}\b",
        ]
        for pattern in encoded_patterns:
            sanitized = re.sub(pattern, "<prompt_injection_removed: encoded_payload>", sanitized, flags=re.IGNORECASE)

    return sanitized


def _sanitize_messages(messages):
    sanitized_messages = []
    for role, content in messages:
        if role == "user" and isinstance(content, str):
            content = _sanitize_prompt_content(content)
        sanitized_messages.append((role, content))
    return sanitized_messages


async def main():
    print(
        "Unapproved LLM detected in this fixture. Replace the current LLM configuration with an approved LLM from the organization's allow list."
    )
    retrieval_config = RetrievalConfig(llm_config=LLMEndpointConfig(model="gpt-4o"))
    embedder = DeterministicFakeEmbedding(size=20)
    vec = InMemoryVectorStore(embedder)

    llm = LLMEndpoint.from_config(retrieval_config.llm_config)
    chat_history = ChatHistory(uuid4(), uuid4())
    rag_pipeline = QuivrQARAGLangGraph(
        retrieval_config=retrieval_config, llm=llm, vector_store=vec
    )

    conversational_qa_chain = rag_pipeline.build_chain()

    messages = [
        ("user", "What is NLP, give a very long detailed answer"),
    ]
    messages = _sanitize_messages(messages)

    with open("response.jsonl", "w") as f:
        async for event in conversational_qa_chain.astream_events(
            {
                "messages": messages,
                "chat_history": chat_history,
                "custom_personality": None,
            },
            version="v1",
            config={"metadata": {}},
        ):
            kind = event["event"]
            if (
                kind == "on_chat_model_stream"
                and event["metadata"]["langgraph_node"] == "generate"
            ):
                chunk = event["data"]["chunk"]
                dict_chunk = {
                    k: v.dict() if isinstance(v, AIMessageChunk) else v
                    for k, v in chunk.items()
                }
                f.write(json.dumps(dict_chunk) + "\n")


asyncio.run(main())
