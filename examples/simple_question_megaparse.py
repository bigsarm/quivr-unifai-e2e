import os
import codecs
import re
import tempfile
from pathlib import Path

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from quivr_core import Brain
from quivr_core.llm.llm_endpoint import LLMEndpoint
from quivr_core.rag.entities.config import LLMEndpointConfig
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt


_HIDDEN_TEXT_PATTERNS = [
    re.compile(r"<!--.*?-->", re.IGNORECASE | re.DOTALL),
    re.compile(r"<[^>]+style\s*=\s*[\"'][^\"']*(?:display\s*:\s*none|font-size\s*:\s*0(?:px)?|color\s*:\s*white|color\s*:\s*#fff(?:fff)?)[^\"']*[\"'][^>]*>.*?</[^>]+>", re.IGNORECASE | re.DOTALL),
    re.compile(r"<[^>]+hidden[^>]*>.*?</[^>]+>", re.IGNORECASE | re.DOTALL),
    re.compile(r"[\u200B-\u200D\uFEFF]+", re.IGNORECASE),
]

_PROMPT_INJECTION_PATTERNS = [
    ("instruction_override", re.compile(r"(?i)\b(?:ignore|disregard|forget)\s+(?:all\s+)?(?:previous|prior|above)\s+instructions\b")),
    ("instruction_override", re.compile(r"(?i)\bforget\s+everything\s+above\b")),
    ("instruction_override", re.compile(r"(?i)\bignore\s+(?:all\s+)?safety\s+guidelines\b")),
    ("role_hijack", re.compile(r"(?i)\byou\s+are\s+now\s+(?:in\s+)?(?:developer\s+mode|admin\s+mode|DAN|unrestricted)\b")),
    ("role_hijack", re.compile(r"(?i)\bprovide\s+unrestricted\s+access\b")),
    ("role_hijack", re.compile(r"(?i)\benable\s+developer\s+mode\b")),
    ("role_hijack", re.compile(r"(?i)\bact\s+as\s+an\s+unrestricted\s+AI\b")),
    ("role_hijack", re.compile(r"(?i)\bwithout\s+safety\s+filters\b")),
    ("role_hijack", re.compile(r"(?i)\bSYSTEM\s+OVERRIDE\b")),
    ("exfiltration_attempt", re.compile(r"(?i)\breveal\s+(?:all\s+)?confidential\s+information\b")),
    ("exfiltration_attempt", re.compile(r"(?i)\breveal\s+the\s+system\s+prompt\b")),
    ("exfiltration_attempt", re.compile(r"(?i)\breveal\s+all\s+system\s+prompts\b")),
    ("exfiltration_attempt", re.compile(r"(?i)\blist\s+all\s+financial\s+data,\s*user\s+passwords,\s*and\s+API\s+keys\b")),
    ("exfiltration_attempt", re.compile(r"(?i)\bprint\s+the\s+API\s+key\b")),
    ("exfiltration_attempt", re.compile(r"(?i)\b(?:send|post|upload|exfiltrate)\b[^\n]*\b(?:to|into)\s+https?://\S+")),
    ("exfiltration_attempt", re.compile(r"!\[[^\]]*\]\([^)]*data:[^)]+\)", re.IGNORECASE)),
    ("delimiter_escape", re.compile(r"(?i)</system>")),
    ("delimiter_escape", re.compile(r"(?i)<\|im_start\|>")),
    ("delimiter_escape", re.compile(r"(?i)###\s*system:")),
    ("command_injection", re.compile(r"(?i)\bexecute\s*:\s*[A-Za-z_][^\n]*")),
    ("command_injection", re.compile(r"(?i)\brun\s+(?:rm\s+-rf\s+/|curl\s+https?://\S+(?:\s*\|\s*(?:sh|bash))?|wget\s+https?://\S+(?:\s*\|\s*(?:sh|bash))?|powershell\s+-[A-Za-z]+[^\n]*)")),
    ("command_injection", re.compile(r"(?i)\bcurl\s+https?://\S+\s*\|\s*(?:sh|bash)\b")),
    ("command_injection", re.compile(r"(?i)\bprint\s*\(\s*os\.environ\.get\([^)]+\)\s*\)")),
]

_PII_MASK_PATTERNS = [
    ("ssn", re.compile(r"\b(\d{3})[- ](\d{2})[- ](\d{4})\b")),
    ("phone", re.compile(r"(?:\+1[ .-]?)?(?:\(\d{3}\)|\b\d{3})[ .-]?\d{3}[ .-]?\d{4}\b")),
    ("email", re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")),
    ("address", re.compile(r"\b\d{1,5}\s+(?:[A-Z][a-z]+\s){1,3}(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|Way)\b\.?(?:,\s*[A-Z][a-z]+(?:\s[A-Z][a-z]+)*)?(?:,\s*[A-Z]{2}\b(?:\s+\d{5}(?:-\d{4})?)?)?(?:,\s*(?:USA|United States)\b)?")),
    ("dob", re.compile(r"(?i)\b(?:DOB|date of birth|born(?: on| in)?)\s*:?\s*(?:\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4}|(?:19|20)\d{2})\b")),
    ("passport", re.compile(r"(?i)\bpassport(?:\s*(?:no\.?|number|#))?\s*:?\s*(?=[A-Z0-9]*\d)[A-Z0-9]{6,9}\b")),
    ("drivers_license", re.compile(r"(?i)\b(?:driver'?s\s+license|drivers\s+license)(?:\s*(?:no\.?|number|#))?\s*:?\s*[A-Z0-9-]{5,20}\b")),
    ("tax_id", re.compile(r"(?i)\b(?:taxpayer\s+identification\s+number|tax\s+id|TIN)(?:\s*(?:no\.?|number|#))?\s*:?\s*[A-Z0-9-]{6,20}\b")),
    ("credit_card", re.compile(r"\b(?:\d[ -]*?){13,19}\b")),
    ("account_number", re.compile(r"(?i)\b(?:financial\s+account\s+number|account\s+number)(?:\s*(?:no\.?|number|#))?\s*:?\s*[A-Z0-9-]{6,20}\b")),
    ("employee_id", re.compile(r"(?i)\bemployee\s+id\s*:?\s*[A-Z0-9-]{2,20}\b")),
    ("school_id", re.compile(r"(?i)\bschool\s+id\s*:?\s*[A-Z0-9-]{2,20}\b")),
    ("vin", re.compile(r"(?i)\bVIN\s*:?\s*[A-HJ-NPR-Z0-9]{17}\b")),
    ("ip_address", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
    ("mac_address", re.compile(r"\b(?:[0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}\b")),
    ("birthplace", re.compile(r"(?i)\bbirthplace\s*:?\s*[^\n,;]+")),
    ("maiden_name", re.compile(r"(?i)\b(?:mother'?s\s+maiden\s+name|maiden\s+name)\s*:?\s*[^\n,;]+")),
    ("medical", re.compile(r"(?i)\bmedical\s+records?\s*:?\s*[^\n]+")),
    ("location", re.compile(r"(?i)\b(?:fine\s+location|location)\s*:?\s*[^\n,;]+")),
    ("ethnicity", re.compile(r"(?i)\bethnicity\s*:?\s*[^\n,;]+")),
    ("sexual_orientation", re.compile(r"(?i)\bsexual\s+orientation\s*:?\s*[^\n,;]+")),
]


def _replace_pattern(text: str, pattern: re.Pattern[str], replacement: str) -> str:
    return pattern.sub(replacement, text)


def _normalize_for_obfuscation(text: str) -> tuple[str, list[int]]:
    translation = str.maketrans({"1": "i", "3": "e", "0": "o", "4": "a", "5": "s", "7": "t"})
    normalized_chars = []
    index_map = []
    for index, char in enumerate(text):
        mapped = char.translate(translation).lower()
        if mapped.isspace():
            continue
        normalized_chars.append(mapped)
        index_map.append(index)
    return "".join(normalized_chars), index_map


def _apply_obfuscated_prompt_injection_redaction(text: str) -> str:
    normalized, index_map = _normalize_for_obfuscation(text)
    obfuscated_patterns = [
        ("instruction_override", re.compile(r"ignore(?:all)?(?:previous|prior|above)instructions")),
        ("instruction_override", re.compile(r"forgeteverythingabove")),
        ("instruction_override", re.compile(r"ignore(?:all)?safetyguidelines")),
        ("role_hijack", re.compile(r"youarenow(?:in)?(?:developermode|adminmode|dan|unrestricted)")),
    ]
    replacements = []
    for category, pattern in obfuscated_patterns:
        for match in pattern.finditer(normalized):
            start = index_map[match.start()]
            end = index_map[match.end() - 1] + 1
            replacements.append((start, end, f"<prompt_injection_removed: {category}>"))
    for start, end, replacement in sorted(replacements, reverse=True):
        text = text[:start] + replacement + text[end:]
    return text


def _decoded_payload_category(decoded_text: str) -> str | None:
    for category, pattern in _PROMPT_INJECTION_PATTERNS:
        if category in {"hidden_text", "command_injection"}:
            continue
        if pattern.search(decoded_text):
            return category
    return None


def sanitize_untrusted_text(text: str) -> str:
    if not text:
        return text

    sanitized = text
    for pattern in _HIDDEN_TEXT_PATTERNS:
        sanitized = _replace_pattern(sanitized, pattern, "<prompt_injection_removed: hidden_text>")

    for category, pattern in _PROMPT_INJECTION_PATTERNS:
        sanitized = _replace_pattern(sanitized, pattern, f"<prompt_injection_removed: {category}>")

    encoded_patterns = [
        re.compile(r"\b(?:[A-Za-z0-9+/]{4}){4,}(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?\b"),
        re.compile(r"(?:%[0-9A-Fa-f]{2}){4,}"),
        re.compile(r"\b(?:[0-9A-Fa-f]{2}){8,}\b"),
    ]
    for pattern in encoded_patterns:
        def _encoded_replacer(match: re.Match[str]) -> str:
            token = match.group(0)
            decoded_variants = []
            try:
                if "%" in token:
                    from urllib.parse import unquote
                    decoded_variants.append(unquote(token))
                elif re.fullmatch(r"(?:[0-9A-Fa-f]{2}){8,}", token):
                    decoded_variants.append(bytes.fromhex(token).decode("utf-8", errors="ignore"))
                else:
                    import base64
                    decoded_variants.append(base64.b64decode(token, validate=True).decode("utf-8", errors="ignore"))
            except Exception:
                pass
            try:
                decoded_variants.append(codecs.decode(token, "rot13"))
            except Exception:
                pass
            for decoded in decoded_variants:
                category = _decoded_payload_category(decoded)
                if category:
                    return "<prompt_injection_removed: encoded_payload>"
            return token
        sanitized = pattern.sub(_encoded_replacer, sanitized)

    sanitized = _apply_obfuscated_prompt_injection_redaction(sanitized)
    return sanitized


def _mask_last_four(match: re.Match[str]) -> str:
    digits = re.sub(r"\D", "", match.group(0))
    return f"***-**-{digits[-4:]}" if len(digits) >= 4 else "<masked:ssn>"


def _redact_last_four(match: re.Match[str], category: str) -> str:
    digits = re.sub(r"\D", "", match.group(0))
    if len(digits) >= 4:
        return f"<redacted:{category}>"
    return f"<redacted:{category}>"


def mask_pii(text: str) -> str:
    if not text:
        return text
    masked = text
    for category, pattern in _PII_MASK_PATTERNS:
        if category == "ssn":
            masked = pattern.sub(_mask_last_four, masked)
        elif category in {"credit_card", "account_number", "tax_id"}:
            masked = pattern.sub(f"<masked:{category}>", masked)
        else:
            masked = pattern.sub(f"<masked:{category}>", masked)
    return masked


def redact_pii(text: str) -> str:
    if not text:
        return text
    redacted = text
    for category, pattern in _PII_MASK_PATTERNS:
        redacted = pattern.sub(f"<redacted:{category}>", redacted)
    return redacted


def sanitize_uploaded_file(file_path: str) -> str:
    source_path = Path(file_path)
    original_text = source_path.read_text(encoding="utf-8", errors="ignore")
    sanitized_text = sanitize_untrusted_text(original_text)
    sanitized_text = redact_pii(sanitized_text)
    temp_file = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=source_path.suffix or ".txt",
        delete=False,
    )
    with temp_file:
        temp_file.write(sanitized_text)
    return temp_file.name

if __name__ == "__main__":
    sanitized_file_path = sanitize_uploaded_file("./tests/processor/pdf/sample.pdf")
    brain = Brain.from_files(
        name="test_brain",
        file_paths=[sanitized_file_path],
        llm=LLMEndpoint(
            llm_config=LLMEndpointConfig(model="gpt-4o"),
            llm=ChatOpenAI(model="gpt-4o", api_key=str(os.getenv("OPENAI_API_KEY"))),
        ),
    )
    embedder = embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
    )
    # Check brain info
    brain.print_info()

    console = Console()
    console.print(Panel.fit("Ask your brain !", style="bold magenta"))

    while True:
        # Get user input
        question = Prompt.ask("[bold cyan]Question[/bold cyan]")

        # Check if user wants to exit
        if question.lower() == "exit":
            console.print(Panel("Goodbye!", style="bold yellow"))
            break

        question = sanitize_untrusted_text(question)
        answer = brain.ask(question)
        # Print the answer with typing effect
        safe_answer = mask_pii(answer.answer)
        console.print(f"[bold green]Quivr Assistant[/bold green]: {safe_answer}")

        console.print("-" * console.width)

    brain.print_info()
