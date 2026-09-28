import os
import base64
import re
import tempfile
from pathlib import Path
from urllib.parse import unquote

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from quivr_core import Brain
from quivr_core.llm.llm_endpoint import LLMEndpoint
from quivr_core.rag.entities.config import LLMEndpointConfig
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt


def _decode_text_payloads(text: str) -> list[str]:
    decoded_payloads: list[str] = []
    if not isinstance(text, str) or not text:
        return decoded_payloads

    url_decoded = unquote(text)
    if url_decoded != text:
        decoded_payloads.append(url_decoded)

    for candidate in re.findall(r"\b(?:[A-Za-z0-9+/]{20,}={0,2})\b", text):
        try:
            decoded = base64.b64decode(candidate, validate=True).decode("utf-8", errors="ignore")
        except Exception:
            continue
        if decoded:
            decoded_payloads.append(decoded)

    return decoded_payloads


def _looks_like_leetspeak_instruction(text: str) -> bool:
    if not isinstance(text, str) or not text:
        return False
    normalized = text.lower().translate(str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s"}))
    return any(
        phrase in normalized
        for phrase in (
            "ignore previous instructions",
            "forget everything above",
            "developer mode",
            "act as unrestricted",
            "you are now dan",
        )
    )


def _has_injection_signal(text: str) -> bool:
    if not isinstance(text, str) or not text:
        return False

    checks = (
        r"(?i)\bignore\s+previous\s+instructions\b",
        r"(?i)\bforget\s+everything\s+above\b",
        r"(?i)\byou\s+are\s+now\s+dan\b",
        r"(?i)\bact\s+as\s+unrestricted\b",
        r"(?i)\bdeveloper\s+mode\b",
        r"(?i)</system>|<system>|\[/?system\]",
        r"(?i)<!--(?:(?!-->).){0,400}(ignore|reveal|send|leak|bypass)(?:(?!-->).){0,400}-->",
        r"(?i)\b(system|assistant|tool)\s*:\s*(ignore|reveal|send|leak|bypass)",
        r"(?i)\b(send|post|upload|exfiltrate)\b.{0,80}\bhttps?://\S+",
        r"(?i)\b(reveal|leak|print|show|list)\b.{0,80}\b(system prompt|api keys?|passwords?|secrets?|confidential information)\b",
        r"(?i)\b(?:rm\s+-rf\s+/|curl\s+https?://\S+\s*\|\s*sh|wget\s+https?://\S+\s*\|\s*sh|powershell\s+-enc\b|bash\s+-c\b|python\s+-c\b|subprocess\.(?:run|popen)\b|os\.system\b|exec\(|eval\()",
        r"(?i)i\s*g\s*n\s*o\s*r\s*e\s+p\s*r\s*e\s*v\s*i\s*o\s*u\s*s\s+i\s*n\s*s\s*t\s*r\s*u\s*c\s*t\s*i\s*o\s*n\s*s",
    )

    if any(re.search(pattern, text) for pattern in checks):
        return True

    if _looks_like_leetspeak_instruction(text):
        return True

    for decoded in _decode_text_payloads(text):
        if any(
            re.search(pattern, decoded)
            for pattern in (
                r"(?i)\bignore\s+previous\s+instructions\b",
                r"(?i)\bforget\s+everything\s+above\b",
                r"(?i)\byou\s+are\s+now\s+dan\b",
                r"(?i)\bact\s+as\s+unrestricted\b",
                r"(?i)\bdeveloper\s+mode\b",
                r"(?i)\b(send|post|upload|exfiltrate)\b.{0,80}\bhttps?://\S+",
                r"(?i)\b(?:rm\s+-rf\s+/|curl\s+https?://\S+\s*\|\s*sh|wget\s+https?://\S+\s*\|\s*sh|powershell\s+-enc\b|bash\s+-c\b|python\s+-c\b|subprocess\.(?:run|popen)\b|os\.system\b|exec\(|eval\()",
            )
        ):
            return True

    return False


def _neutralize_prompt_injection(text: str) -> str:
    if not isinstance(text, str) or not text:
        return text

    sanitized = text
    replacements = (
        (r"(?i)\b(ignore\s+previous\s+instructions|forget\s+everything\s+above)\b", "<prompt_injection_removed: instruction_override>"),
        (r"(?i)\b(you\s+are\s+now\s+dan|act\s+as\s+unrestricted)\b", "<prompt_injection_removed: role_hijack>"),
        (r"(?i)(</system>|<system>|\[/?system\]|\n\s*(?:---|===)\s*\n)", "<prompt_injection_removed: delimiter_escape>"),
        (r"(?i)<!--(?:(?!-->).){0,400}(ignore|reveal|send|leak|bypass)(?:(?!-->).){0,400}-->", "<prompt_injection_removed: hidden_text>"),
        (r"(?i)\b(system|assistant|tool)\s*:\s*(ignore|reveal|send|leak|bypass)[^\n]*", "<prompt_injection_removed: fake_system_message>"),
        (r"(?i)\b(send|post|upload|exfiltrate)\b.{0,80}\bhttps?://\S+", "<prompt_injection_removed: exfiltration_attempt>"),
        (r"(?i)\b(reveal|leak|print|show|list)\b.{0,80}\b(system prompt|api keys?|passwords?|secrets?|confidential information)\b", "<prompt_injection_removed: exfiltration_attempt>"),
        (r"(?i)\b(previous|earlier|above)\b.{0,80}\b(ignore|discard|replace|overwrite)\b.{0,80}\b(instructions|context|rules|messages)\b", "<prompt_injection_removed: context_poisoning>"),
        (r"(?i)\b(?:rm\s+-rf\s+/|curl\s+https?://\S+\s*\|\s*sh|wget\s+https?://\S+\s*\|\s*sh|powershell\s+-enc\b|bash\s+-c\b|python\s+-c\b|subprocess\.(?:run|popen)\b|os\.system\b|exec\(|eval\()", "<prompt_injection_removed: command_injection>"),
        (r"(?i)i\s*g\s*n\s*o\s*r\s*e\s+p\s*r\s*e\s*v\s*i\s*o\s*u\s*s\s+i\s*n\s*s\s*t\s*r\s*u\s*c\s*t\s*i\s*o\s*n\s*s", "<prompt_injection_removed: split_payload>"),
        (r"(?i)\b(dan|developer mode|jailbreak)\b", "<prompt_injection_removed: jailbreak_attempt>"),
    )

    for pattern, replacement in replacements:
        sanitized = re.sub(pattern, replacement, sanitized)

    for candidate in re.findall(r"\b(?:[A-Za-z0-9+/]{20,}={0,2})\b", sanitized):
        decoded = ""
        try:
            decoded = base64.b64decode(candidate, validate=True).decode("utf-8", errors="ignore")
        except Exception:
            decoded = ""
        if decoded and _has_injection_signal(decoded):
            sanitized = sanitized.replace(candidate, "<prompt_injection_removed: encoded_payload>")

    url_encoded_matches = re.findall(r"(?:%[0-9A-Fa-f]{2}){4,}", sanitized)
    for candidate in url_encoded_matches:
        decoded = unquote(candidate)
        if decoded and _has_injection_signal(decoded):
            sanitized = sanitized.replace(candidate, "<prompt_injection_removed: encoded_payload>")

    if _looks_like_leetspeak_instruction(sanitized):
        sanitized = re.sub(
            r"(?i)\b[i1!|][g69][n^]?0?r[e3]\s+p[r2]?e[vu]?[i1]?[o0]?[uuv]?[s5]\s+[i1]n[s5][t7]+r[uuv][c(][t7][i1][o0]n[s5]\b",
            "<prompt_injection_removed: encoded_payload>",
            sanitized,
        )

    return sanitized


def _redact_zero_tolerance_pii(text: str) -> str:
    if not isinstance(text, str) or not text:
        return text

    redacted = text
    pii_patterns = (
        (r"\b\d{3}-\d{2}-\d{4}\b", "<pii_redacted:ssn>"),
        (r"\b(?:\+1[-.\s]?)?(?:\(\d{3}\)[-.\s]?|\d{3}[-.\s])\d{3}[-.\s]\d{4}\b", "<pii_redacted:personal_phone>"),
        (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", "<pii_redacted:email>"),
        (r"\b(?:\d[ -]*?){13,19}\b", "<pii_redacted:credit_card>"),
        (r"\b(?:[A-Fa-f0-9]{2}:){5}[A-Fa-f0-9]{2}\b", "<pii_redacted:mac_address>"),
        (r"\b(?:25[0-5]|2[0-4]\d|1?\d?\d)(?:\.(?:25[0-5]|2[0-4]\d|1?\d?\d)){3}\b", "<pii_redacted:ip_address>"),
        (r"\b[A-HJ-NPR-Z0-9]{17}\b", "<pii_redacted:vin>"),
    )
    for pattern, replacement in pii_patterns:
        redacted = re.sub(pattern, replacement, redacted)

    labeled_patterns = (
        (r"(?im)(\bDOB\b\s*[:\-]\s*)([^\n]+)", "<pii_redacted:year_of_birth>"),
        (r"(?im)(\bYear of Birth\b\s*[:\-]\s*)([^\n]+)", "<pii_redacted:year_of_birth>"),
        (r"(?im)(\bBirthplace\b\s*[:\-]\s*)([^\n]+)", "<pii_redacted:birthplace>"),
        (r"(?im)(\bMother'?s Maiden Name\b\s*[:\-]\s*)([^\n]+)", "<pii_redacted:mothers_maiden_name>"),
        (r"(?im)(\bHome Address\b\s*[:\-]\s*)([^\n]+)", "<pii_redacted:home_address>"),
        (r"(?im)(\bPassport(?: Number| No\.?| #)?\b\s*[:\-]\s*)([^\n]+)", "<pii_redacted:passport_number>"),
        (r"(?im)(\bDriver'?s License(?: Number| No\.?| #)?\b\s*[:\-]\s*)([^\n]+)", "<pii_redacted:drivers_license_number>"),
        (r"(?im)(\bTaxpayer Identification Number\b\s*[:\-]\s*)([^\n]+)", "<pii_redacted:taxpayer_identification_number>"),
        (r"(?im)(\bFinancial Account Number\b\s*[:\-]\s*)([^\n]+)", "<pii_redacted:financial_account_number>"),
        (r"(?im)(\bEmployee Id\b\s*[:\-]\s*)([^\n]+)", "<pii_redacted:employee_id>"),
        (r"(?im)(\bSchool Id\b\s*[:\-]\s*)([^\n]+)", "<pii_redacted:school_id>"),
        (r"(?im)(\bFine Location\b\s*[:\-]\s*)([^\n]+)", "<pii_redacted:fine_location>"),
        (r"(?im)(\bEthnicity\b\s*[:\-]\s*)([^\n]+)", "<pii_redacted:ethnicity>"),
        (r"(?im)(\bSexual Orientation\b\s*[:\-]\s*)([^\n]+)", "<pii_redacted:sexual_orientation>"),
        (r"(?im)(\bMedical Records\b\s*[:\-]\s*)([^\n]+)", "<pii_redacted:medical_records>"),
        (r"(?im)(\bFingerprints\b\s*[:\-]\s*)([^\n]+)", "<pii_redacted:fingerprints>"),
        (r"(?im)(\bRetina/Iris Scan\b\s*[:\-]\s*)([^\n]+)", "<pii_redacted:retina_iris_scan>"),
        (r"(?im)(\bVoice signature\b\s*[:\-]\s*)([^\n]+)", "<pii_redacted:voice_signature>"),
        (r"(?im)(\bFacial image\b\s*[:\-]\s*)([^\n]+)", "<pii_redacted:facial_image>"),
    )
    for pattern, replacement in labeled_patterns:
        redacted = re.sub(pattern, lambda m: f"{m.group(1)}{replacement}", redacted)

    return redacted


def _prepare_uploaded_file(file_path: str) -> str:
    source_path = Path(file_path)
    file_bytes = source_path.read_bytes()
    text = file_bytes.decode("utf-8", errors="ignore")
    sanitized_text = _neutralize_prompt_injection(_redact_zero_tolerance_pii(text))
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=source_path.suffix)
    with open(temp_file.name, "w", encoding="utf-8") as sanitized_file:
        sanitized_file.write(sanitized_text)
    return temp_file.name


def _sanitize_prompt(text: str) -> str:
    return _neutralize_prompt_injection(text)

if __name__ == "__main__":
    uploaded_file_path = "./tests/processor/pdf/sample.pdf"
    sanitized_file_path = _prepare_uploaded_file(uploaded_file_path)
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
    console.print(Panel("Configured model 'gpt-4o' must be replaced with an approved organization model before production use.", style="bold red"))

    while True:
        # Get user input
        question = Prompt.ask("[bold cyan]Question[/bold cyan]")
        question = _sanitize_prompt(question)

        # Check if user wants to exit
        if question.lower() == "exit":
            console.print(Panel("Goodbye!", style="bold yellow"))
            break

        answer = brain.ask(question)
        # Print the answer with typing effect
        console.print(f"[bold green]Quivr Assistant[/bold green]: {answer.answer}")

        console.print("-" * console.width)

    brain.print_info()
