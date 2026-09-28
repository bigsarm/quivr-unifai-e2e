from uuid import uuid4
import base64
import binascii
import re
import urllib.parse

import pytest
from quivr_core.rag.entities.chat import ChatHistory
from quivr_core.rag.entities.config import LLMEndpointConfig, RetrievalConfig
from quivr_core.llm import LLMEndpoint
from quivr_core.rag.entities.models import ParsedRAGChunkResponse, RAGResponseMetadata
from quivr_core.rag.quivr_rag_langgraph import QuivrQARAGLangGraph


_ZERO_WIDTH_RE = re.compile(r"[\u200B-\u200D\u2060\uFEFF]")
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.IGNORECASE | re.DOTALL)
_HIDDEN_STYLE_RE = re.compile(
    r"<(?:span|div|p)[^>]*style\s*=\s*['\"][^'\"]*(?:display\s*:\s*none|font-size\s*:\s*0(?:px)?|color\s*:\s*white)[^'\"]*['\"][^>]*>.*?</(?:span|div|p)>",
    re.IGNORECASE | re.DOTALL,
)
_BASE64_TOKEN_RE = re.compile(r"\b(?:[A-Za-z0-9+/]{4}){8,}(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?\b")
_HEX_TOKEN_RE = re.compile(r"\b(?:0x)?(?:[0-9a-fA-F]{2}){8,}\b")
_URL_ENCODED_RE = re.compile(r"(?:%[0-9a-fA-F]{2}){4,}")
_SPLIT_IGNORE_RE = re.compile(r"i\s*g\s*n\s*o\s*r\s*e\s+p\s*r\s*e\s*v\s*i\s*o\s*u\s*s\s+i\s*n\s*s\s*t\s*r\s*u\s*c\s*t\s*i\s*o\s*n\s*s", re.IGNORECASE)


def _looks_like_leetspeak_instruction(text: str) -> bool:
    lowered = text.lower()
    translated = lowered.translate(str.maketrans({"1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "0": "o", "@": "a", "$": "s"}))
    return any(
        phrase in translated
        for phrase in (
            "ignore previous instructions",
            "forget everything above",
            "act as unrestricted",
            "you are now dan",
            "developer mode",
        )
    )


def _decode_base64_if_instruction(token: str) -> str | None:
    try:
        decoded = base64.b64decode(token, validate=True).decode("utf-8", errors="ignore")
    except (binascii.Error, ValueError):
        return None
    lowered = decoded.lower()
    if any(
        phrase in lowered
        for phrase in (
            "ignore previous instructions",
            "forget everything above",
            "act as unrestricted",
            "you are now dan",
            "developer mode",
            "reveal system prompt",
            "curl http://",
            "wget http://",
            "rm -rf",
        )
    ):
        return decoded
    return None


def sanitize_prompt_content(text: str) -> str:
    sanitized = text
    sanitized = _HTML_COMMENT_RE.sub("<prompt_injection_removed: hidden_text>", sanitized)
    sanitized = _HIDDEN_STYLE_RE.sub("<prompt_injection_removed: hidden_text>", sanitized)
    sanitized = _ZERO_WIDTH_RE.sub("<prompt_injection_removed: hidden_text>", sanitized)

    if _looks_like_leetspeak_instruction(sanitized):
        sanitized = "<prompt_injection_removed: encoded_payload>"

    instruction_override_patterns = (
        re.compile(r"\bignore\s+previous\s+instructions\b", re.IGNORECASE),
        re.compile(r"\bforget\s+everything\s+above\b", re.IGNORECASE),
    )
    for pattern in instruction_override_patterns:
        sanitized = pattern.sub("<prompt_injection_removed: instruction_override>", sanitized)
    sanitized = _SPLIT_IGNORE_RE.sub("<prompt_injection_removed: split_payload>", sanitized)

    role_hijack_patterns = (
        re.compile(r"\byou\s+are\s+now\s+dan\b", re.IGNORECASE),
        re.compile(r"\bact\s+as\s+unrestricted\b", re.IGNORECASE),
    )
    for pattern in role_hijack_patterns:
        sanitized = pattern.sub("<prompt_injection_removed: role_hijack>", sanitized)

    jailbreak_patterns = (
        re.compile(r"\bdeveloper\s+mode\b", re.IGNORECASE),
        re.compile(r"\bdo\s+anything\s+now\b", re.IGNORECASE),
        re.compile(r"\bfictional\s+framing\s+bypass\b", re.IGNORECASE),
    )
    for pattern in jailbreak_patterns:
        sanitized = pattern.sub("<prompt_injection_removed: jailbreak_attempt>", sanitized)

    delimiter_patterns = (
        re.compile(r"</system>", re.IGNORECASE),
        re.compile(r"<system>", re.IGNORECASE),
        re.compile(r"\n\s*---\s*\n"),
        re.compile(r"\n\s*===\s*\n"),
    )
    for pattern in delimiter_patterns:
        sanitized = pattern.sub("<prompt_injection_removed: delimiter_escape>", sanitized)

    fake_system_patterns = (
        re.compile(r"\bsystem\s*:\s*you\s+must\b", re.IGNORECASE),
        re.compile(r"\btool\s*:\s*return\b", re.IGNORECASE),
    )
    for pattern in fake_system_patterns:
        sanitized = pattern.sub("<prompt_injection_removed: fake_system_message>", sanitized)

    exfiltration_patterns = (
        re.compile(r"\breveal\s+system\s+prompt\b", re.IGNORECASE),
        re.compile(r"!\[[^\]]*\]\([^)]*https?://[^)]*\)", re.IGNORECASE),
        re.compile(r"\b(?:send|post|upload|exfiltrate)\b.{0,40}\bhttps?://\S+", re.IGNORECASE),
    )
    for pattern in exfiltration_patterns:
        sanitized = pattern.sub("<prompt_injection_removed: exfiltration_attempt>", sanitized)

    context_poisoning_patterns = (
        re.compile(r"\bin\s+the\s+next\s+message\b", re.IGNORECASE),
        re.compile(r"\bfrom\s+now\s+on\b", re.IGNORECASE),
    )
    for pattern in context_poisoning_patterns:
        sanitized = pattern.sub("<prompt_injection_removed: context_poisoning>", sanitized)

    indirect_injection_patterns = (
        re.compile(r"\b(?:file|metadata|comment)\s*:\s*ignore\s+previous\s+instructions\b", re.IGNORECASE),
    )
    for pattern in indirect_injection_patterns:
        sanitized = pattern.sub("<prompt_injection_removed: indirect_injection>", sanitized)

    command_patterns = (
        re.compile(r"\bcurl\s+https?://\S+", re.IGNORECASE),
        re.compile(r"\bwget\s+https?://\S+", re.IGNORECASE),
        re.compile(r"\brm\s+-rf\b", re.IGNORECASE),
        re.compile(r"\b(?:bash|sh|powershell|cmd)\b\s+-[A-Za-z]\s+", re.IGNORECASE),
        re.compile(r"\b(?:os\.system|subprocess\.(?:run|popen|call)|eval|exec)\s*\(", re.IGNORECASE),
    )
    for pattern in command_patterns:
        sanitized = pattern.sub("<prompt_injection_removed: command_injection>", sanitized)

    for match in list(_BASE64_TOKEN_RE.finditer(sanitized)):
        token = match.group(0)
        if _decode_base64_if_instruction(token) is not None:
            sanitized = sanitized.replace(token, "<prompt_injection_removed: encoded_payload>")

    for match in list(_HEX_TOKEN_RE.finditer(sanitized)):
        token = match.group(0)
        try:
            raw = token[2:] if token.lower().startswith("0x") else token
            decoded = bytes.fromhex(raw).decode("utf-8", errors="ignore").lower()
        except ValueError:
            continue
        if any(
            phrase in decoded
            for phrase in (
                "ignore previous instructions",
                "forget everything above",
                "act as unrestricted",
                "you are now dan",
                "curl http://",
                "rm -rf",
            )
        ):
            sanitized = sanitized.replace(token, "<prompt_injection_removed: encoded_payload>")

    for match in list(_URL_ENCODED_RE.finditer(sanitized)):
        token = match.group(0)
        decoded = urllib.parse.unquote(token).lower()
        if any(
            phrase in decoded
            for phrase in (
                "ignore previous instructions",
                "forget everything above",
                "act as unrestricted",
                "you are now dan",
                "curl http://",
                "rm -rf",
            )
        ):
            sanitized = sanitized.replace(token, "<prompt_injection_removed: encoded_payload>")

    return sanitized


@pytest.fixture(scope="function")
def mock_chain_qa_stream(monkeypatch, chunks_stream_answer):
    class MockQAChain:
        async def astream_events(self, *args, **kwargs):
            default_metadata = {
                "langgraph_node": "generate",
                "is_final_node": False,
                "citations": None,
                "followup_questions": None,
                "sources": None,
                "metadata_model": None,
            }

            # Send all chunks except the last one
            for chunk in chunks_stream_answer[:-1]:
                yield {
                    "event": "on_chat_model_stream",
                    "metadata": default_metadata,
                    "data": {"chunk": chunk["answer"]},
                }

            # Send the last chunk
            yield {
                "event": "end",
                "metadata": {
                    "langgraph_node": "generate",
                    "is_final_node": True,
                    "citations": [],
                    "followup_questions": None,
                    "sources": [],
                    "metadata_model": None,
                },
                "data": {"chunk": chunks_stream_answer[-1]["answer"]},
            }

    def mock_qa_chain(*args, **kwargs):
        self = args[0]
        self.final_nodes = ["generate"]
        return MockQAChain()

    monkeypatch.setattr(QuivrQARAGLangGraph, "build_chain", mock_qa_chain)


@pytest.mark.base
@pytest.mark.asyncio
async def test_quivrqaraglanggraph(
    mem_vector_store, full_response, mock_chain_qa_stream, openai_api_key
):
    pytest.fail(
        "Replace the unapproved LLM/agent usage in this test (GPT model via LLMEndpoint) with an organization-approved LLM from the registry allow list."
    )
    # Making sure the model
    llm_config = LLMEndpointConfig(model="gpt-4o")
    llm = LLMEndpoint.from_config(llm_config)
    retrieval_config = RetrievalConfig(llm_config=llm_config)
    chat_history = ChatHistory(uuid4(), uuid4())
    rag_pipeline = QuivrQARAGLangGraph(
        retrieval_config=retrieval_config, llm=llm, vector_store=mem_vector_store
    )

    stream_responses: list[ParsedRAGChunkResponse] = []

    # Making sure that we are calling the func_calling code path
    assert rag_pipeline.llm_endpoint.supports_func_calling()
    prompt = sanitize_prompt_content("answer in bullet points. tell me something")
    async for resp in rag_pipeline.answer_astream(
        prompt, chat_history, []
    ):
        stream_responses.append(resp)

    # This assertion passed
    assert all(
        not r.last_chunk for r in stream_responses[:-1]
    ), "Some chunks before last have last_chunk=True"
    assert stream_responses[-1].last_chunk

    # Let's check this assertion
    for idx, response in enumerate(stream_responses[1:-1]):
        assert (
            len(response.answer) > 0
        ), f"Sent an empty answer {response} at index {idx+1}"

    # Verify metadata
    default_metadata = RAGResponseMetadata().model_dump()
    assert all(
        r.metadata.model_dump() == default_metadata for r in stream_responses[:-1]
    )
    last_response = stream_responses[-1]
    # TODO(@aminediro) : test responses with sources
    assert last_response.metadata.sources == []
    assert last_response.metadata.citations == []

    # Assert whole response makes sense
    assert "".join([r.answer for r in stream_responses]) == full_response
