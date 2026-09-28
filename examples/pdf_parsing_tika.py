from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models import FakeListChatModel
from quivr_core import Brain
from quivr_core.rag.entities.config import LLMEndpointConfig
from quivr_core.llm.llm_endpoint import LLMEndpoint
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
import base64
import codecs
import re
import urllib.parse


def _replace_spans(text, spans):
    if not spans:
        return text
    spans = sorted(spans, key=lambda item: item[0])
    merged = []
    for start, end, replacement in spans:
        if not merged or start >= merged[-1][1]:
            merged.append([start, end, replacement])
        else:
            merged[-1][1] = max(merged[-1][1], end)
            merged[-1][2] = replacement
    parts = []
    cursor = 0
    for start, end, replacement in merged:
        parts.append(text[cursor:start])
        parts.append(replacement)
        cursor = end
    parts.append(text[cursor:])
    return "".join(parts)


def _normalized_with_mapping(text):
    translation = str.maketrans({"1": "i", "3": "e", "0": "o", "4": "a", "5": "s", "7": "t"})
    normalized_chars = []
    mapping = []
    index = 0
    while index < len(text):
        char = text[index]
        if char.isspace():
            while index < len(text) and text[index].isspace():
                index += 1
            continue
        normalized_chars.append(char.translate(translation).lower())
        mapping.append(index)
        index += 1
    return "".join(normalized_chars), mapping


def _find_obfuscated_prompt_spans(text):
    normalized_text, mapping = _normalized_with_mapping(text)
    spans = []
    patterns = [
        (r"ignore(?:all)?(?:previous|prior|above)instructions", "instruction_override"),
        (r"forgeteverythingabove", "instruction_override"),
        (r"ignore(?:all)?safetyguidelines", "instruction_override"),
        (r"youarenow(?:in)?developermode", "role_hijack"),
        (r"youarenow(?:in)?adminmode", "role_hijack"),
        (r"youarenowdan", "role_hijack"),
        (r"youarenowunrestricted", "role_hijack"),
        (r"provideunrestrictedaccess", "role_hijack"),
        (r"enabledevelopermode", "role_hijack"),
        (r"actasanunrestrictedai", "role_hijack"),
        (r"withoutsafetyfilters", "role_hijack"),
        (r"systemoverride", "role_hijack"),
    ]
    for pattern, category in patterns:
        for match in re.finditer(pattern, normalized_text, re.IGNORECASE):
            start = mapping[match.start()]
            end = mapping[match.end() - 1] + 1
            spans.append((start, end, f"<prompt_injection_removed: {category}>"))
    return spans


def _decoded_attack_category(decoded_text):
    checks = [
        (r"(?i)\b(?:ignore|disregard|forget)\s+(?:all\s+)?(?:previous|prior|above)\s+instructions\b", "instruction_override"),
        (r"(?i)\bforget\s+everything\s+above\b", "instruction_override"),
        (r"(?i)\bignore\s+(?:all\s+)?safety\s+guidelines\b", "instruction_override"),
        (r"(?i)\byou\s+are\s+now\s+(?:in\s+)?(?:developer\s+mode|admin\s+mode|DAN|unrestricted)\b", "role_hijack"),
        (r"(?i)\b(?:provide\s+unrestricted\s+access|enable\s+developer\s+mode|act\s+as\s+an\s+unrestricted\s+AI|without\s+safety\s+filters|SYSTEM\s+OVERRIDE)\b", "role_hijack"),
        (r"(?i)\b(?:reveal\s+(?:all\s+)?confidential\s+information|reveal\s+the\s+system\s+prompt|reveal\s+all\s+system\s+prompts|list\s+all\s+financial\s+data,\s*user\s+passwords,\s*and\s+API\s+keys|print\s+the\s+API\s+key|send\s+data\s+to\s+https?://\S+)\b", "exfiltration_attempt"),
        (r"(?i)(?:</system>|<\|im_start\|>|###\s*system:)", "delimiter_escape"),
        (r"(?i)\b(?:execute|run)\s+(?:[:]\s*)?(?:print\([^\n]+\)|rm\s+-rf\s+\S+|powershell\s+\S+|bash\s+-c\s+\S+|sh\s+-c\s+\S+|python\s+-c\s+\S+|curl\s+https?://\S+(?:\s*\|\s*(?:sh|bash))?)", "command_injection"),
    ]
    for pattern, category in checks:
        if re.search(pattern, decoded_text):
            return category
    return None


def sanitize_untrusted_text(text):
    if not isinstance(text, str) or not text:
        return text

    sanitized = text

    pii_patterns = [
        (r"\b\d{3}[- ]\d{2}[- ]\d{4}\b", "ssn"),
        (r"(?:\+1[ .-]?)?(?:\(\d{3}\)|\b\d{3})[ .-]?\d{3}[ .-]?\d{4}\b", "phone"),
        (r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b", "email"),
        (r"\b\d{1,5}\s+(?:[A-Z][a-z]+\s){1,3}(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|Way)\b\.?(?:,\s*[A-Z][a-z]+(?:\s[A-Z][a-z]+)*)?(?:,\s*[A-Z]{2}\b(?:\s+\d{5}(?:-\d{4})?)?)?(?:,\s*(?:USA|United States)\b)?", "address"),
        (r"(?i)\b(?:DOB|date of birth|born(?: on| in)?)\s*:?\s*(?:\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4}|(?:19|20)\d{2})\b", "dob"),
        (r"(?i)\bpassport(?:\s*(?:no\.?|number|#))?\s*:?\s*(?=[A-Z0-9]*\d)[A-Z0-9]{6,9}\b", "passport"),
        (r"(?i)\b(?:driver'?s license|drivers license|driver license|dl)(?:\s*(?:no\.?|number|#))?\s*:?\s*[A-Z0-9-]{4,20}\b", "drivers_license"),
        (r"(?i)\b(?:taxpayer identification number|tax identification number|TIN)(?:\s*(?:no\.?|number|#))?\s*:?\s*[A-Z0-9-]{6,20}\b", "tax_id"),
        (r"\b(?:\d[ -]*?){13,19}\b", "credit_card"),
        (r"(?i)\b(?:financial account number|account number)(?:\s*(?:no\.?|number|#))?\s*:?\s*[A-Z0-9-]{6,34}\b", "account_number"),
        (r"(?i)\bemployee\s*id\s*:?\s*[A-Z0-9-]{2,20}\b", "employee_id"),
        (r"(?i)\bschool\s*id\s*:?\s*[A-Z0-9-]{2,20}\b", "school_id"),
        (r"(?i)\bVIN\s*:?\s*[A-HJ-NPR-Z0-9]{17}\b", "vin"),
        (r"\b(?:\d{1,3}\.){3}\d{1,3}\b", "ip_address"),
        (r"\b(?:[0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}\b", "mac_address"),
        (r"(?i)\b(?:birthplace|place of birth)\s*:?\s*[^\n,;]+", "birthplace"),
        (r"(?i)\b(?:mother'?s maiden name|maiden name)\s*:?\s*[^\n,;]+", "maiden_name"),
        (r"(?i)\bmedical records?\s*:?\s*[^\n]+", "medical"),
        (r"(?i)\b(?:fine location|location)\s*:?\s*[^\n,;]+", "location"),
        (r"(?i)\bethnicity\s*:?\s*[^\n,;]+", "ethnicity"),
        (r"(?i)\bsexual orientation\s*:?\s*[^\n,;]+", "sexual_orientation"),
    ]
    for pattern, label in pii_patterns:
        sanitized = re.sub(pattern, f"<redacted:{label}>", sanitized)

    prompt_patterns = [
        (r"<!--.*?-->", "hidden_text"),
        (r"(?is)<(?P<tag>\w+)[^>]*style\s*=\s*['\"][^'\"]*(?:display\s*:\s*none|font-size\s*:\s*0(?:px)?|color\s*:\s*white)[^'\"]*['\"][^>]*>.*?</(?P=tag)>", "hidden_text"),
        (r"[\u200B-\u200D\uFEFF]+", "hidden_text"),
        (r"(?i)\b(?:ignore|disregard|forget)\s+(?:all\s+)?(?:previous|prior|above)\s+instructions\b", "instruction_override"),
        (r"(?i)\bforget\s+everything\s+above\b", "instruction_override"),
        (r"(?i)\bignore\s+(?:all\s+)?safety\s+guidelines\b", "instruction_override"),
        (r"(?i)\byou\s+are\s+now\s+(?:in\s+)?(?:developer\s+mode|admin\s+mode|DAN|unrestricted)\b", "role_hijack"),
        (r"(?i)\b(?:provide\s+unrestricted\s+access|enable\s+developer\s+mode|act\s+as\s+an\s+unrestricted\s+AI|without\s+safety\s+filters|SYSTEM\s+OVERRIDE)\b", "role_hijack"),
        (r"(?i)\b(?:reveal\s+(?:all\s+)?confidential\s+information|reveal\s+the\s+system\s+prompt|reveal\s+all\s+system\s+prompts|list\s+all\s+financial\s+data,\s*user\s+passwords,\s*and\s+API\s+keys|print\s+the\s+API\s+key|send\s+data\s+to\s+https?://\S+)\b", "exfiltration_attempt"),
        (r"!\[[^\]]*\]\([^)]*data:[^)]*\)", "exfiltration_attempt"),
        (r"(?i)(?:</system>|<\|im_start\|>|###\s*system:)", "delimiter_escape"),
        (r"(?i)\b(?:execute|run)\s+(?:[:]\s*)?(?:print\([^\n]+\)|rm\s+-rf\s+\S+|powershell\s+\S+|bash\s+-c\s+\S+|sh\s+-c\s+\S+|python\s+-c\s+\S+|curl\s+https?://\S+(?:\s*\|\s*(?:sh|bash))?)", "command_injection"),
    ]
    for pattern, category in prompt_patterns:
        sanitized = re.sub(pattern, f"<prompt_injection_removed: {category}>", sanitized)

    spans = []
    encoded_patterns = [
        r"\b(?:[A-Za-z0-9+/]{4}){4,}(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?\b",
        r"(?:%[0-9A-Fa-f]{2}){4,}",
        r"\b(?:[0-9A-Fa-f]{2}){8,}\b",
    ]
    for pattern in encoded_patterns:
        for match in re.finditer(pattern, sanitized):
            candidate = match.group(0)
            decoded_values = []
            if "%" in candidate:
                try:
                    decoded_values.append(urllib.parse.unquote(candidate))
                except Exception:
                    pass
            if re.fullmatch(r"\b(?:[0-9A-Fa-f]{2}){8,}\b", candidate):
                try:
                    decoded_values.append(bytes.fromhex(candidate).decode("utf-8", errors="ignore"))
                except Exception:
                    pass
            if re.fullmatch(r"\b(?:[A-Za-z0-9+/]{4}){4,}(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?\b", candidate):
                try:
                    decoded_values.append(base64.b64decode(candidate, validate=True).decode("utf-8", errors="ignore"))
                except Exception:
                    pass
            for decoded in decoded_values:
                if _decoded_attack_category(decoded):
                    spans.append((match.start(), match.end(), "<prompt_injection_removed: encoded_payload>"))
                    break

    try:
        rot13_text = codecs.decode(sanitized, "rot13")
    except Exception:
        rot13_text = ""
    rot13_category = _decoded_attack_category(rot13_text) if rot13_text else None
    if rot13_category:
        spans.append((0, len(sanitized), "<prompt_injection_removed: encoded_payload>"))

    spans.extend(_find_obfuscated_prompt_spans(sanitized))
    sanitized = _replace_spans(sanitized, spans)
    return sanitized


def _patch_tika_parsing():
    try:
        from quivr_core.processor.implementations import tika_processor
    except Exception:
        return

    if getattr(tika_processor, "_sanitizer_patched", False):
        return

    original_parser = getattr(tika_processor.parser, "from_file", None)
    if original_parser is None:
        return

    def _sanitized_from_file(*args, **kwargs):
        parsed = original_parser(*args, **kwargs)
        if isinstance(parsed, dict) and isinstance(parsed.get("content"), str):
            parsed["content"] = sanitize_untrusted_text(parsed["content"])
        return parsed

    tika_processor.parser.from_file = _sanitized_from_file
    tika_processor._sanitizer_patched = True


_patch_tika_parsing()

if __name__ == "__main__":
    brain = Brain.from_files(
        name="test_brain",
        file_paths=["tests/processor/data/dummy.pdf"],
        llm=LLMEndpoint(
            llm=FakeListChatModel(responses=["good"]),
            llm_config=LLMEndpointConfig(model="fake_model", llm_base_url="local"),
        ),
        embedder=DeterministicFakeEmbedding(size=20),
    )
    # Check brain info
    brain.print_info()

    console = Console()
    console.print(Panel.fit("Ask your brain !", style="bold magenta"))

    while True:
        # Get user input
        question = Prompt.ask("[bold cyan]Question[/bold cyan]")
        question = sanitize_untrusted_text(question)

        # Check if user wants to exit
        if question.lower() == "exit":
            console.print(Panel("Goodbye!", style="bold yellow"))
            break

        answer = brain.ask(question)
        if hasattr(answer, "answer") and isinstance(answer.answer, str):
            answer.answer = sanitize_untrusted_text(answer.answer)
        # Print the answer with typing effect
        console.print(f"[bold green]Quivr Assistant[/bold green]: {answer.answer}")

        console.print("-" * console.width)

    brain.print_info()
