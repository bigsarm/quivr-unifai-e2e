from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models import FakeListChatModel
from quivr_core import Brain
from quivr_core.rag.entities.config import LLMEndpointConfig
from quivr_core.llm.llm_endpoint import LLMEndpoint
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
import base64
import io
import re
from pathlib import Path

try:
    from pypdf import PdfReader, PdfWriter
except ImportError:
    PdfReader = None
    PdfWriter = None


_DISAPPROVED_MODELS = {
    "deepseekchat",
    "deepseekr1",
    "deepseekr1distillllama70b",
    "deepseekreasoner",
    "customllmclientnull",
    "openrouternull",
    "usdeepseekr1v10null",
}


def _normalize_model_name(value: str) -> str:
    return re.sub(r"[\s\-_.:]", "", value or "").lower()


def _ensure_model_allowed(model_name: str) -> None:
    normalized = _normalize_model_name(model_name)
    if normalized in _DISAPPROVED_MODELS:
        raise ValueError(
            "Configured LLM is not approved. Replace it with an approved LLM from the organization's allow list."
        )


def _try_decode_base64_text(value: str) -> str | None:
    compact = re.sub(r"\s+", "", value)
    if len(compact) < 16 or len(compact) % 4 != 0 or not re.fullmatch(r"[A-Za-z0-9+/=]+", compact):
        return None
    try:
        decoded = base64.b64decode(compact, validate=True)
        text = decoded.decode("utf-8")
    except Exception:
        return None
    if any(ord(ch) < 32 and ch not in "\n\r\t" for ch in text):
        return None
    return text


def _normalize_leetspeak(value: str) -> str:
    translation = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s"})
    return value.translate(translation)


def _sanitize_prompt(value: str) -> str:
    sanitized = value

    hidden_patterns = [
        (r"<!--.*?-->", "<prompt_injection_removed: hidden_text>"),
        (r"[\u200B-\u200F\u2060\uFEFF]+", "<prompt_injection_removed: hidden_text>"),
        (r"<[^>]+style\s*=\s*[\"'][^\"']*(display\s*:\s*none|visibility\s*:\s*hidden|font-size\s*:\s*0|color\s*:\s*white)[^\"']*[\"'][^>]*>.*?</[^>]+>", "<prompt_injection_removed: hidden_text>"),
    ]
    for pattern, replacement in hidden_patterns:
        sanitized = re.sub(pattern, replacement, sanitized, flags=re.IGNORECASE | re.DOTALL)

    direct_patterns = [
        (r"\b(ignore previous instructions|ignore all previous instructions|forget everything above|disregard earlier instructions)\b", "<prompt_injection_removed: instruction_override>"),
        (r"\b(you are now dan|act as unrestricted|developer mode|jailbreak|do anything now|bypass safety)\b", "<prompt_injection_removed: jailbreak_attempt>"),
        (r"\b(act as|you are now)\b", "<prompt_injection_removed: role_hijack>"),
        (r"</?(system|assistant|tool|developer)>|```+(system|assistant|tool|developer)?", "<prompt_injection_removed: delimiter_escape>"),
        (r"\b(system prompt|reveal your instructions|send data to https?://|upload .*https?://|exfiltrat(e|ion)|leak .*prompt)\b", "<prompt_injection_removed: exfiltration_attempt>"),
        (r"\b(in future turns|from now on|persist this instruction|remember this rule)\b", "<prompt_injection_removed: context_poisoning>"),
        (r"\b(system:|assistant:|tool:)\b", "<prompt_injection_removed: fake_system_message>"),
        (r"\b(eval\s*\(|exec\s*\(|os\.system\s*\(|subprocess\.(run|popen|call)\s*\(|curl\b|wget\b|bash\b|sh\b|cmd\.exe\b|powershell\b|chmod\b|base64\s+-d\b)", "<prompt_injection_removed: command_injection>"),
        (r"\b(instruction[s]?\s*[:=-].*|comment[s]?\s*[:=-].*)\b", "<prompt_injection_removed: indirect_injection>"),
        (r"i\s*g\s*n\s*o\s*r\s*e|d\s*a\s*n", "<prompt_injection_removed: split_payload>"),
    ]
    for pattern, replacement in direct_patterns:
        sanitized = re.sub(pattern, replacement, sanitized, flags=re.IGNORECASE)

    decoded = _try_decode_base64_text(sanitized)
    if decoded:
        decoded_lower = _normalize_leetspeak(decoded.lower())
        if re.search(r"\b(ignore previous instructions|forget everything above|you are now|act as|system prompt|bash|powershell|curl|wget|eval|exec)\b", decoded_lower):
            sanitized = re.sub(re.escape(sanitized), "<prompt_injection_removed: encoded_payload>", sanitized, count=1)

    normalized_leet = _normalize_leetspeak(sanitized.lower())
    if normalized_leet != sanitized.lower() and re.search(r"\b(ignore previous instructions|forget everything above|you are now|act as unrestricted|developer mode|jailbreak)\b", normalized_leet):
        sanitized = "<prompt_injection_removed: encoded_payload>"

    if re.search(r"\b(MZ|ELF)\b", sanitized):
        sanitized = re.sub(r"\b(MZ|ELF)\b", "<prompt_injection_removed: command_injection>", sanitized)

    return sanitized


def _extract_pdf_text(file_path: str) -> str:
    if PdfReader is None:
        raise RuntimeError("PDF text inspection requires pypdf to be installed.")
    reader = PdfReader(file_path)
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def _redact_pii(text: str) -> tuple[str, bool]:
    redacted = text
    pii_patterns = [
        (r"\b\d{3}-\d{2}-\d{4}\b", "<redacted:ssn>"),
        (r"\b(?:19|20)\d{2}\b", "<redacted:year_of_birth>"),
        (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", "<redacted:email>"),
        (r"\b(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?){2}\d{4}\b", "<redacted:personal_phone_number>"),
        (r"\b\d{13,19}\b", "<redacted:credit_card_or_financial_account_number>"),
        (r"\b[A-Z]{1,2}\d{6,9}\b", "<redacted:passport_or_id_number>"),
        (r"\b(?:\d{1,5}\s+[A-Za-z0-9.\-\s]+\s(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct))\b", "<redacted:home_address>"),
        (r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b", "<redacted:ip_address>"),
        (r"\b(?:[0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}\b", "<redacted:mac_address>"),
        (r"\b[A-HJ-NPR-Z0-9]{17}\b", "<redacted:vin>"),
    ]
    for pattern, replacement in pii_patterns:
        redacted = re.sub(pattern, replacement, redacted)
    return redacted, redacted != text


def _inspect_uploaded_file_for_pii(file_path: str) -> None:
    text = _extract_pdf_text(file_path)
    _, changed = _redact_pii(text)
    if changed:
        raise ValueError(
            f"Uploaded file '{Path(file_path).name}' contains zero-tolerance PII. Redact the file contents before ingestion."
        )

if __name__ == "__main__":
    file_paths = ["tests/processor/data/dummy.pdf"]
    for file_path in file_paths:
        _inspect_uploaded_file_for_pii(file_path)
    _ensure_model_allowed("fake_model")
    brain = Brain.from_files(
        name="test_brain",
        file_paths=file_paths,
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

        # Check if user wants to exit
        if question.lower() == "exit":
            console.print(Panel("Goodbye!", style="bold yellow"))
            break

        question = _sanitize_prompt(question)
        answer = brain.ask(question)
        # Print the answer with typing effect
        console.print(f"[bold green]Quivr Assistant[/bold green]: {answer.answer}")

        console.print("-" * console.width)

    brain.print_info()
