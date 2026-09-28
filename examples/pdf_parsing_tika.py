from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models import FakeListChatModel
from quivr_core import Brain
from quivr_core.rag.entities.config import LLMEndpointConfig
from quivr_core.llm.llm_endpoint import LLMEndpoint
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
import re
import sys
import urllib.parse
from base64 import b64decode


def _normalize_for_detection(text):
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _looks_like_base64_payload(token):
    if len(token) < 16 or len(token) % 4 != 0:
        return False
    if not re.fullmatch(r"[A-Za-z0-9+/=]+", token):
        return False
    try:
        decoded = b64decode(token, validate=True).decode("utf-8", errors="ignore")
    except Exception:
        return False
    normalized = _normalize_for_detection(decoded)
    suspicious_markers = (
        "ignorepreviousinstructions",
        "forgeteverythingabove",
        "actasunrestricted",
        "revealallsystemprompts",
        "curlhttp",
        "wgethttp",
        "bash-c",
        "powershell",
        "subprocess",
    )
    return any(marker in normalized for marker in suspicious_markers)


def sanitize_prompt(question):
    sanitized = question
    replacements = [
        (re.compile(r"(?i)\b(ignore\s+previous\s+instructions|forget\s+everything\s+above)\b"), "<prompt_injection_removed: instruction_override>"),
        (re.compile(r"(?i)\b(you\s+are\s+now\s+dan|act\s+as\s+an?\s+unrestricted\b[^\n]*|developer\s+mode|dan\s+mode|jailbreak)\b"), "<prompt_injection_removed: role_hijack>"),
        (re.compile(r"(?is)</?system>|</?assistant>|</?tool>|(^|\n)\s*(---|===)\s*(\n|$)"), "<prompt_injection_removed: delimiter_escape>"),
        (re.compile(r"(?is)<!--.*?(ignore\s+previous\s+instructions|forget\s+everything\s+above|act\s+as\s+an?\s+unrestricted).*?-->"), "<prompt_injection_removed: hidden_text>"),
        (re.compile(r"[\u200b-\u200f\ufeff]+"), "<prompt_injection_removed: hidden_text>"),
        (re.compile(r"(?i)\b(system\s*:\s*|tool\s*:\s*|assistant\s*:\s*)(ignore\s+previous\s+instructions|reveal\s+all\s+confidential\s+information|list\s+all\s+passwords\s+and\s+api\s+keys)\b"), "<prompt_injection_removed: fake_system_message>"),
        (re.compile(r"(?i)\b(send|post|upload|exfiltrate|leak)\b[^\n]*\b(to|via)\b[^\n]*(https?://\S+|ftp://\S+)"), "<prompt_injection_removed: exfiltration_attempt>"),
        (re.compile(r"(?i)\b(reveal|show|print|list)\b[^\n]*\b(system\s+prompt|passwords?|api\s+keys?|confidential\s+information)\b"), "<prompt_injection_removed: exfiltration_attempt>"),
        (re.compile(r"(?i)\b(on\s+your\s+next\s+reply|in\s+all\s+future\s+responses|from\s+now\s+on)\b"), "<prompt_injection_removed: context_poisoning>"),
        (re.compile(r"(?i)\b(metadata|comment|code\s+comment|data\s+field|filename)\b[^\n]*\b(ignore\s+previous\s+instructions|act\s+as\s+an?\s+unrestricted)\b"), "<prompt_injection_removed: indirect_injection>"),
        (re.compile(r"(?i)\b(curl\s+https?://\S+|wget\s+https?://\S+|bash\s+-c\s+\S+|sh\s+-c\s+\S+|powershell(?:\.exe)?\s+-[A-Za-z]+\s+\S+|cmd(?:\.exe)?\s+/c\s+\S+|python\s+-c\s+\S+|subprocess\.(?:run|Popen)\s*\(|os\.system\s*\(|eval\s*\(|exec\s*\()"), "<prompt_injection_removed: command_injection>"),
        (re.compile(r"(?i)i\s*g\s*n\s*o\s*r\s*e\s+p\s*r\s*e\s*v\s*i\s*o\s*u\s*s\s+i\s*n\s*s\s*t\s*r\s*u\s*c\s*t\s*i\s*o\s*n\s*s"), "<prompt_injection_removed: split_payload>"),
        (re.compile(r"(?i)\b(dan|developer\s+mode|do\s+anything\s+now|fictional\s+framing)\b"), "<prompt_injection_removed: jailbreak_attempt>"),
    ]
    for pattern, replacement in replacements:
        sanitized = pattern.sub(replacement, sanitized)

    for token in re.findall(r"\b[A-Za-z0-9+/=]{16,}\b", sanitized):
        if _looks_like_base64_payload(token):
            sanitized = sanitized.replace(token, "<prompt_injection_removed: encoded_payload>")

    url_decoded = urllib.parse.unquote(sanitized)
    if url_decoded != sanitized:
        normalized_decoded = _normalize_for_detection(url_decoded)
        if any(marker in normalized_decoded for marker in ("ignorepreviousinstructions", "forgeteverythingabove", "actasunrestricted", "youarenowdan")):
            sanitized = "<prompt_injection_removed: encoded_payload>"

    normalized = _normalize_for_detection(sanitized)
    if any(marker in normalized for marker in ("1gn0r3pr3v10us1nstruct10ns", "y0uaren0wdan", "actasunrestricted")):
        sanitized = "<prompt_injection_removed: encoded_payload>"

    if re.search(r"(?i)<[^>]+style\s*=\s*[\"'][^\"']*(display\s*:\s*none|font-size\s*:\s*0|color\s*:\s*white)[^\"']*[\"'][^>]*>.*?(ignore\s+previous\s+instructions|forget\s+everything\s+above|act\s+as\s+an?\s+unrestricted).*?</[^>]+>", question):
        sanitized = "<prompt_injection_removed: hidden_text>"

    return sanitized

if __name__ == "__main__":
    print(
        "Configured model 'fake_model' is not approved. Replace it with an approved organizational LLM before running this example.",
        file=sys.stderr,
    )
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
        question = sanitize_prompt(question)

        # Check if user wants to exit
        if question.lower() == "exit":
            console.print(Panel("Goodbye!", style="bold yellow"))
            break

        answer = brain.ask(question)
        # Print the answer with typing effect
        console.print(f"[bold green]Quivr Assistant[/bold green]: {answer.answer}")

        console.print("-" * console.width)

    brain.print_info()
