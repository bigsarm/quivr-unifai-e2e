import logging
import base64
import binascii
import re
from typing import Any, Dict, List, Tuple, no_type_check

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.messages.ai import AIMessageChunk
from langchain_core.prompts import format_document
from langfuse.callback import CallbackHandler

from quivr_core.rag.entities.config import WorkflowConfig
from quivr_core.rag.entities.models import (
    ChatLLMMetadata,
    ParsedRAGResponse,
    QuivrKnowledge,
    RAGResponseMetadata,
    RawRAGResponse,
)
from quivr_core.rag.prompts import TemplatePromptName, custom_prompts

# TODO(@aminediro): define a types packages where we clearly define IO types
# This should be used for serialization/deseriallization later


logger = logging.getLogger("quivr_core")


_DISAPPROVED_MODEL_NORMALIZED = {
    "deepseekchat",
    "deepseekr1",
    "deepseekr1distillllama70b",
    "deepseekreasoner",
    "customllmclientnull",
    "deepseekchatnull",
    "openrouternull",
    "usdeepseekr1v10null",
}

_ZERO_WIDTH_TRANSLATION = str.maketrans("", "", "\u200b\u200c\u200d\ufeff\u2060")


def _normalize_model_identifier(model_name: str) -> str:
    return re.sub(r"[\s\-_.:]", "", model_name).casefold()


def _ensure_model_is_approved(model_name: str) -> str:
    normalized_model_name = _normalize_model_identifier(model_name)
    if normalized_model_name in _DISAPPROVED_MODEL_NORMALIZED:
        raise ValueError(
            "Unapproved LLM configured. Replace it with an approved LLM from the organization's allow list."
        )
    return model_name


def _looks_like_base64_payload(text: str) -> bool:
    compact = re.sub(r"\s+", "", text)
    if len(compact) < 24 or len(compact) % 4 != 0:
        return False
    if not re.fullmatch(r"[A-Za-z0-9+/=]+", compact):
        return False
    try:
        decoded = base64.b64decode(compact, validate=True).decode("utf-8", errors="ignore")
    except (binascii.Error, ValueError):
        return False
    return _contains_encoded_instruction(decoded)


def _contains_encoded_instruction(decoded_text: str) -> bool:
    lowered = decoded_text.casefold()
    encoded_instruction_patterns = [
        r"\bignore\s+previous\s+instructions\b",
        r"\bforget\s+everything\s+above\b",
        r"\byou\s+are\s+now\b",
        r"\bact\s+as\s+an?\s+unrestricted\b",
        r"\breveal\s+(?:the\s+)?system\s+prompt\b",
        r"\bcurl\s+https?://",
        r"\bwget\s+https?://",
        r"\bos\.system\s*\(",
        r"\bsubprocess\.(?:run|Popen|call)\s*\(",
        r"\b(?:rm|chmod|chown|powershell|cmd(?:\.exe)?|bash|sh)\b[^\n]{0,80}\b(?:-c|/c)\b",
    ]
    return any(re.search(pattern, lowered) for pattern in encoded_instruction_patterns)


def _sanitize_prompt_injection_content(text: str, for_document: bool = False) -> str:
    sanitized_text = text

    hidden_patterns = [
        (
            re.compile(r"<!--(?:(?!-->).)*(?:ignore\s+previous\s+instructions|forget\s+everything\s+above|you\s+are\s+now|act\s+as\s+an?\s+unrestricted)(?:(?!-->).)*-->", re.IGNORECASE | re.DOTALL),
            "<prompt_injection_removed: hidden_text>",
        ),
        (
            re.compile(r"<[^>]*style\s*=\s*[\"'][^\"']*(?:display\s*:\s*none|font-size\s*:\s*0|color\s*:\s*(?:#fff(?:fff)?|white))[^\"']*[\"'][^>]*>.*?</[^>]+>", re.IGNORECASE | re.DOTALL),
            "<prompt_injection_removed: hidden_text>",
        ),
        (
            re.compile(r"(?:[\u200b\u200c\u200d\ufeff\u2060]{3,}).{0,120}(?:ignore\s+previous\s+instructions|forget\s+everything\s+above|you\s+are\s+now)", re.IGNORECASE | re.DOTALL),
            "<prompt_injection_removed: hidden_text>",
        ),
    ]
    for pattern, marker in hidden_patterns:
        sanitized_text = pattern.sub(marker, sanitized_text)

    encoded_patterns = [
        (
            re.compile(r"\b(?:[A-Fa-f0-9]{2}\s+){8,}[A-Fa-f0-9]{2}\b"),
            "<prompt_injection_removed: encoded_payload>",
        ),
        (
            re.compile(r"(?:%[0-9A-Fa-f]{2}){6,}"),
            "<prompt_injection_removed: encoded_payload>",
        ),
        (
            re.compile(r"(?:\\u[0-9A-Fa-f]{4}){4,}"),
            "<prompt_injection_removed: encoded_payload>",
        ),
        (
            re.compile(r"\b(?:[.-]{1,6}\s+){6,}[.-]{1,6}\b"),
            "<prompt_injection_removed: encoded_payload>",
        ),
        (
            re.compile(r"\b1g[nm]0r[e3]\s+pr[e3]v[i1]0[u\"]?s\s+[i1]nstr[u\"]ct[i1]0ns\b", re.IGNORECASE),
            "<prompt_injection_removed: encoded_payload>",
        ),
    ]
    for pattern, marker in encoded_patterns:
        sanitized_text = pattern.sub(marker, sanitized_text)

    if _looks_like_base64_payload(sanitized_text):
        sanitized_text = re.sub(
            r"\b(?:[A-Za-z0-9+/]{20,}={0,2})\b",
            "<prompt_injection_removed: encoded_payload>",
            sanitized_text,
        )

    replacement_patterns = [
        (
            re.compile(r"\b(?:ignore\s+previous\s+instructions|forget\s+everything\s+above|disregard\s+all\s+prior\s+directions)\b", re.IGNORECASE),
            "<prompt_injection_removed: instruction_override>",
        ),
        (
            re.compile(r"\b(?:you\s+are\s+now\s+(?:dan|developer\s+mode|admin\s+mode)|act\s+as\s+an?\s+unrestricted\s+ai|pretend\s+to\s+be\s+an?\s+unrestricted\s+ai)\b", re.IGNORECASE),
            "<prompt_injection_removed: role_hijack>",
        ),
        (
            re.compile(r"(?:</system>|</assistant>|<system>|<assistant>|(?:\n|^)\s*(?:---|===)\s*(?:\n|$)){1,}", re.IGNORECASE),
            "<prompt_injection_removed: delimiter_escape>",
        ),
        (
            re.compile(r"\b(?:system\s*:\s*|assistant\s*:\s*|tool\s*:\s*)(?:ignore\s+previous\s+instructions|reveal\s+the\s+system\s+prompt|you\s+must\s+comply)\b", re.IGNORECASE),
            "<prompt_injection_removed: fake_system_message>",
        ),
        (
            re.compile(r"(?:!\[[^\]]*\]\(https?://[^)]+\)|\b(?:send|post|upload|exfiltrate|leak)\b[^\n]{0,120}\b(?:https?://\S+|system\s+prompt|passwords?|api\s+keys?)\b)", re.IGNORECASE),
            "<prompt_injection_removed: exfiltration_attempt>",
        ),
        (
            re.compile(r"\b(?:on\s+your\s+next\s+reply\s+say|in\s+every\s+future\s+response|from\s+now\s+on\s+you\s+must|remember\s+this\s+instruction\s+for\s+later)\b", re.IGNORECASE),
            "<prompt_injection_removed: context_poisoning>",
        ),
        (
            re.compile(r"\b(?:run|execute)\b[^\n]{0,80}\b(?:curl\s+https?://\S+|wget\s+https?://\S+|bash\s+-c\b[^\n]*|sh\s+-c\b[^\n]*|powershell(?:\.exe)?\b[^\n]*|-EncodedCommand\b|cmd(?:\.exe)?\s+/c\b[^\n]*|os\.system\s*\([^\n]*|subprocess\.(?:run|Popen|call)\s*\([^\n]*)", re.IGNORECASE),
            "<prompt_injection_removed: command_injection>",
        ),
        (
            re.compile(r"\b(?:DAN|developer\s+mode|jailbreak|fictional\s+framing\s+bypass)\b", re.IGNORECASE),
            "<prompt_injection_removed: jailbreak_attempt>",
        ),
        (
            re.compile(r"\bi\s*\.?\s*g\s*\.?\s*n\s*\.?\s*o\s*\.?\s*r\s*\.?\s*e\b.{0,40}\bp\s*\.?\s*r\s*\.?\s*e\s*\.?\s*v\s*\.?\s*i\s*\.?\s*o\s*\.?\s*u\s*\.?\s*s\b.{0,40}\bi\s*\.?\s*n\s*\.?\s*s\s*\.?\s*t\s*\.?\s*r\s*\.?\s*u\s*\.?\s*c\s*\.?\s*t\s*\.?\s*i\s*\.?\s*o\s*\.?\s*n\s*\.?\s*s\b", re.IGNORECASE),
            "<prompt_injection_removed: split_payload>",
        ),
    ]

    if for_document:
        replacement_patterns.append(
            (
                re.compile(r"(?:#|//|/\*)[^\n]*(?:ignore\s+previous\s+instructions|forget\s+everything\s+above|reveal\s+the\s+system\s+prompt)[^\n]*(?:\*/)?", re.IGNORECASE),
                "<prompt_injection_removed: indirect_injection>",
            )
        )

    for pattern, marker in replacement_patterns:
        sanitized_text = pattern.sub(marker, sanitized_text)

    sanitized_text = sanitized_text.translate(_ZERO_WIDTH_TRANSLATION)
    return sanitized_text


def model_supports_function_calling(model_name: str):
    _ensure_model_is_approved(model_name)
    models_not_supporting_function_calls: list[str] = ["llama2", "test", "ollama3"]

    return model_name not in models_not_supporting_function_calls


def format_history_to_openai_mesages(
    tuple_history: List[Tuple[str, str]], system_message: str, question: str
) -> List[BaseMessage]:
    """Format the chat history into a list of Base Messages"""
    question = _sanitize_prompt_injection_content(question)
    messages = []
    messages.append(SystemMessage(content=system_message))
    for human, ai in tuple_history:
        messages.append(HumanMessage(content=human))
        messages.append(AIMessage(content=ai))
    messages.append(HumanMessage(content=question))
    return messages


def cited_answer_filter(tool):
    return tool["name"] == "cited_answer"


def get_chunk_metadata(
    msg: AIMessageChunk, sources: list[Any] | None = None
) -> RAGResponseMetadata:
    metadata = {"sources": sources or []}

    if not msg.tool_calls:
        return RAGResponseMetadata(**metadata, metadata_model=None)

    all_citations = []
    all_followup_questions = []

    for tool_call in msg.tool_calls:
        if tool_call.get("name") == "cited_answer" and "args" in tool_call:
            args = tool_call["args"]
            all_citations.extend(args.get("citations", []))
            all_followup_questions.extend(args.get("followup_questions", []))

    metadata["citations"] = all_citations
    metadata["followup_questions"] = all_followup_questions[:3]  # Limit to 3

    return RAGResponseMetadata(**metadata, metadata_model=None)


def get_prev_message_str(msg: AIMessageChunk) -> str:
    if msg.tool_calls:
        cited_answer = next(x for x in msg.tool_calls if cited_answer_filter(x))
        if "args" in cited_answer and "answer" in cited_answer["args"]:
            return cited_answer["args"]["answer"]
    return ""


# TODO: CONVOLUTED LOGIC !
# TODO(@aminediro): redo this
@no_type_check
def parse_chunk_response(
    rolling_msg: AIMessageChunk,
    raw_chunk: AIMessageChunk,
    supports_func_calling: bool,
    previous_content: str = "",
) -> Tuple[AIMessageChunk, str, str]:
    """Parse a chunk response
    Args:
        rolling_msg: The accumulated message so far
        raw_chunk: The new chunk to add
        supports_func_calling: Whether function calling is supported
        previous_content: The previous content string
    Returns:
        Tuple of (updated rolling message, new content only, full content)
    """
    rolling_msg += raw_chunk

    tool_calls = rolling_msg.tool_calls

    if not supports_func_calling or not tool_calls:
        new_content = raw_chunk.content  # Just the new chunk's content
        full_content = rolling_msg.content  # The full accumulated content
        return rolling_msg, new_content, full_content

    current_answers = get_answers_from_tool_calls(tool_calls)
    full_answer = "\n\n".join(current_answers)
    if not full_answer:
        full_answer = previous_content

    new_content = full_answer[len(previous_content) :]

    return rolling_msg, new_content, full_answer


def get_answers_from_tool_calls(tool_calls):
    answers = []
    for tool_call in tool_calls:
        if tool_call.get("name") == "cited_answer":
            args = tool_call.get("args", {})
            if isinstance(args, dict):
                answers.append(args.get("answer", ""))
            else:
                logger.warning(f"Expected dict for tool_call args, got {type(args)}")
    return answers


@no_type_check
def parse_response(raw_response: RawRAGResponse, model_name: str) -> ParsedRAGResponse:
    _ensure_model_is_approved(model_name)
    answers = []
    sources = raw_response["docs"] if "docs" in raw_response else []

    metadata = RAGResponseMetadata(
        sources=sources, metadata_model=ChatLLMMetadata(name=model_name)
    )

    if (
        model_supports_function_calling(model_name)
        and "tool_calls" in raw_response["answer"]
        and raw_response["answer"].tool_calls
    ):
        all_citations = []
        all_followup_questions = []
        for tool_call in raw_response["answer"].tool_calls:
            if "args" in tool_call:
                args = tool_call["args"]
                if "citations" in args:
                    all_citations.extend(args["citations"])
                if "followup_questions" in args:
                    all_followup_questions.extend(args["followup_questions"])
                if "answer" in args:
                    answers.append(args["answer"])
        metadata.citations = all_citations
        metadata.followup_questions = all_followup_questions
    else:
        answers.append(raw_response["answer"].content)

    answer_str = "\n".join(answers)
    parsed_response = ParsedRAGResponse(answer=answer_str, metadata=metadata)
    return parsed_response


def combine_documents(
    docs,
    document_prompt=custom_prompts[TemplatePromptName.DEFAULT_DOCUMENT_PROMPT],
    document_separator="\n\n",
):
    # for each docs, add an index in the metadata to be able to cite the sources
    for doc, index in zip(docs, range(len(docs)), strict=False):
        doc.metadata["index"] = index
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    doc_strings = [
        _sanitize_prompt_injection_content(doc_string, for_document=True)
        for doc_string in doc_strings
    ]
    return document_separator.join(doc_strings)


def format_file_list(
    list_files_array: list[QuivrKnowledge], max_files: int = 20
) -> str:
    list_files = [file.file_name or file.url for file in list_files_array]
    files: list[str] = list(filter(lambda n: n is not None, list_files))  # type: ignore
    files = files[:max_files]

    files_str = "\n".join(files) if list_files_array else "None"
    return files_str


def collect_tools(workflow_config: WorkflowConfig):
    validated_tools = "Available tools which can be activated:\n"
    for i, tool in enumerate(workflow_config.validated_tools):
        validated_tools += f"Tool {i+1} name: {tool.name}\n"
        validated_tools += f"Tool {i+1} description: {tool.description}\n\n"

    activated_tools = "Activated tools which can be deactivated:\n"
    for i, tool in enumerate(workflow_config.activated_tools):
        activated_tools += f"Tool {i+1} name: {tool.name}\n"
        activated_tools += f"Tool {i+1} description: {tool.description}\n\n"

    return validated_tools, activated_tools


def format_dict(kv: Dict[str, str]) -> str:
    return "\n".join([f"{k}: {v}" for k, v in kv.items() if v is not None and v != ""])


class LangfuseService:
    def __init__(self):
        self.langfuse_handler = CallbackHandler()

    def get_handler(self):
        return self.langfuse_handler
