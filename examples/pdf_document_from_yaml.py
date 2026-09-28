import asyncio
import logging
import os
import re
import tempfile
from pathlib import Path

try:
    from pypdf import PdfReader
except ImportError:  # pragma: no cover - fallback for environments with legacy package
    from PyPDF2 import PdfReader

import dotenv
from quivr_core import Brain
from quivr_core.rag.entities.config import AssistantConfig
from rich.traceback import install as rich_install

ConsoleOutputHandler = logging.StreamHandler()

logger = logging.getLogger("quivr_core")
logger.setLevel(logging.DEBUG)
logger.addHandler(ConsoleOutputHandler)


logger = logging.getLogger("megaparse")
logger.setLevel(logging.DEBUG)
logger.addHandler(ConsoleOutputHandler)


# Install rich's traceback handler to automatically format tracebacks
rich_install()


def _replace_if_decoded_attack(match: re.Match[str], marker: str) -> str:
    value = match.group(0)
    try:
        decoded = bytes.fromhex(value[2:]).decode("utf-8", errors="ignore")
    except ValueError:
        return value
    if _contains_prompt_attack(decoded):
        return marker
    return value


def _contains_prompt_attack(text: str) -> bool:
    lowered = text.lower()
    attack_checks = [
        r"\bignore\s+previous\s+instructions\b",
        r"\bforget\s+everything\s+(?:above|before)\b",
        r"\byou\s+are\s+now\s+(?:dan|in\s+admin\s+mode)\b",
        r"\bact\s+as\s+(?:an\s+)?unrestricted\b",
        r"\bdeveloper\s+mode\b",
        r"\breveal\s+(?:the\s+)?system\s+prompt\b",
        r"\blist\s+all\s+(?:passwords|api\s+keys)\b",
        r"\bcurl\s+https?://\S+",
        r"\bwget\s+https?://\S+",
        r"\b(?:bash|sh|powershell|cmd(?:\.exe)?)\s+-[cC]\b",
    ]
    return any(re.search(pattern, lowered) for pattern in attack_checks)


def _neutralize_prompt_injection(text: str) -> str:
    sanitized = text
    replacements = [
        (
            r"(?i)\b(?:ignore\s+previous\s+instructions|forget\s+everything\s+(?:above|before)|disregard\s+(?:all\s+)?prior\s+instructions)\b",
            "<prompt_injection_removed: instruction_override>",
        ),
        (
            r"(?i)\b(?:you\s+are\s+now\s+(?:dan|in\s+admin\s+mode)|act\s+as\s+(?:an\s+)?unrestricted(?:\s+ai)?|assume\s+the\s+role\s+of\s+(?:system|developer))\b",
            "<prompt_injection_removed: role_hijack>",
        ),
        (
            r"(?is)</?(?:system|assistant|tool|developer)>|\[/?(?:system|assistant|tool|developer)\]|(?:^|\n)\s*(?:---|===)\s*(?:system|assistant|developer|tool)\s*(?:---|===)",
            "<prompt_injection_removed: delimiter_escape>",
        ),
        (
            r"(?is)<!--.*?(?:ignore\s+previous\s+instructions|act\s+as|reveal\s+the\s+system\s+prompt|list\s+all\s+(?:passwords|api\s+keys)).*?-->",
            "<prompt_injection_removed: hidden_text>",
        ),
        (
            r"[\u200b-\u200f\ufeff]+",
            "<prompt_injection_removed: hidden_text>",
        ),
        (
            r"(?is)<(?:(?:span|div|p)[^>]*style\s*=\s*[\"'][^\"']*(?:display\s*:\s*none|font-size\s*:\s*0|color\s*:\s*white)[^\"']*[\"'][^>]*)>.*?</(?:span|div|p)>",
            "<prompt_injection_removed: hidden_text>",
        ),
        (
            r"(?i)\b(?:system|assistant|tool|developer)\s*:\s*(?:ignore\s+previous\s+instructions|reveal\s+the\s+system\s+prompt|list\s+all\s+(?:passwords|api\s+keys)|you\s+must\s+comply)\b.*",
            "<prompt_injection_removed: fake_system_message>",
        ),
        (
            r"(?i)!\[[^\]]*\]\(https?://[^)]+\)|\b(?:send|post|upload|exfiltrate|leak)\b.{0,80}\b(?:to|into)\b.{0,20}https?://\S+|\b(?:reveal|print|dump|leak)\b.{0,40}\b(?:system\s+prompt|passwords|api\s+keys|confidential\s+information)\b",
            "<prompt_injection_removed: exfiltration_attempt>",
        ),
        (
            r"(?i)\b(?:from\s+now\s+on|in\s+the\s+next\s+turn|for\s+the\s+rest\s+of\s+this\s+chat|persist\s+this\s+instruction)\b.*",
            "<prompt_injection_removed: context_poisoning>",
        ),
        (
            r"(?i)\b(?:metadata|comment|code\s+comment|document\s+field)\s*:\s*(?:ignore\s+previous\s+instructions|act\s+as|reveal\s+the\s+system\s+prompt)\b.*",
            "<prompt_injection_removed: indirect_injection>",
        ),
        (
            r"(?i)\b(?:curl|wget)\s+https?://\S+|\b(?:bash|sh|powershell|cmd(?:\.exe)?)\s+-[cC]\s+.+|\b(?:os\.system|subprocess\.(?:run|Popen)|eval|exec)\s*\(.+",
            "<prompt_injection_removed: command_injection>",
        ),
        (
            r"(?i)\b(?:dan|developer\s+mode|jailbreak|bypass\s+safety|fictional\s+framing)\b.*",
            "<prompt_injection_removed: jailbreak_attempt>",
        ),
    ]
    for pattern, replacement in replacements:
        sanitized = re.sub(pattern, replacement, sanitized)

    sanitized = re.sub(
        r"(?i)(?:[A-Za-z0-9+/]{20,}={0,2})",
        lambda match: "<prompt_injection_removed: encoded_payload>"
        if _contains_prompt_attack(match.group(0))
        else match.group(0),
        sanitized,
    )
    sanitized = re.sub(
        r"(?i)%(?:[0-9a-f]{2}){6,}",
        lambda match: "<prompt_injection_removed: encoded_payload>"
        if _contains_prompt_attack(match.group(0))
        else match.group(0),
        sanitized,
    )
    sanitized = re.sub(
        r"(?i)0x(?:[0-9a-f]{2}){6,}",
        lambda match: _replace_if_decoded_attack(
            match, "<prompt_injection_removed: encoded_payload>"
        ),
        sanitized,
    )
    return sanitized


def _redact_pii(text: str) -> str:
    sanitized = text
    pii_patterns = [
        (r"\b\d{3}-\d{2}-\d{4}\b", "<redacted:ssn>"),
        (r"(?i)\b(?:email|e-mail)\s*:\s*[^\s,;]+@[^\s,;]+", lambda m: m.group(0).split(":", 1)[0] + ": <redacted:email>"),
        (r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[A-Za-z]{2,}\b", "<redacted:email>"),
        (r"(?i)\b(?:phone|telephone|mobile|cell)\s*:\s*(?:\+?\d[\d\s().-]{7,}\d)", lambda m: m.group(0).split(":", 1)[0] + ": <redacted:phone>"),
        (r"(?<!\w)(?:\+\d{1,3}[\s-]?)?(?:\(?\d{2,4}\)?[\s-]?){2,4}\d{2,4}(?!\w)", lambda m: "<redacted:phone>" if re.search(r"\+|\(|\)|-|\s", m.group(0)) and len(re.sub(r"\D", "", m.group(0))) >= 10 else m.group(0)),
        (r"(?i)\baddress\s*:\s*[^\n]+", lambda m: m.group(0).split(":", 1)[0] + ": <redacted:home_address>"),
        (r"(?i)\b(?:birthplace|place\s+of\s+birth)\s*:\s*[^\n]+", lambda m: m.group(0).split(":", 1)[0] + ": <redacted:birthplace>"),
        (r"(?i)\b(?:year\s+of\s+birth|yob|dob)\s*:\s*[^\n]+", lambda m: m.group(0).split(":", 1)[0] + ": <redacted:year_of_birth>"),
        (r"(?i)\bborn\s+in\s+\d{4}\b", "born in <redacted:year_of_birth>"),
        (r"(?i)\bmother(?:'s|s)?\s+maiden\s+name\s*:\s*[^\n]+", lambda m: m.group(0).split(":", 1)[0] + ": <redacted:maiden_name>"),
        (r"(?i)\bpassport(?:\s+(?:number|no\.?))?\s*:\s*[A-Z0-9-]+", lambda m: m.group(0).split(":", 1)[0] + ": <redacted:passport_number>"),
        (r"(?i)\bdriver'?s\s+license(?:\s+number)?\s*:\s*[A-Z0-9-]+", lambda m: m.group(0).split(":", 1)[0] + ": <redacted:drivers_license_number>"),
        (r"(?i)\b(?:tin|taxpayer\s+identification\s+number)\s*:\s*[A-Z0-9-]+", lambda m: m.group(0).split(":", 1)[0] + ": <redacted:taxpayer_identification_number>"),
        (r"(?i)\b(?:employee|school)\s+id\s*:\s*[A-Z0-9-]+", lambda m: m.group(0).split(":", 1)[0] + ": <redacted:id>"),
        (r"(?i)\bvin\s*:\s*[A-HJ-NPR-Z0-9]{17}\b", lambda m: m.group(0).split(":", 1)[0] + ": <redacted:vin>"),
        (r"\b(?:\d[ -]*?){13,19}\b", lambda m: "<redacted:financial_number>" if 13 <= len(re.sub(r"\D", "", m.group(0))) <= 19 else m.group(0)),
        (r"\b(?:\d{1,3}\.){3}\d{1,3}\b", "<redacted:ip_address>"),
        (r"\b(?:[0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}\b", "<redacted:mac_address>"),
        (r"(?i)\b(?:medical\s+record|medical\s+records)\s*:\s*[^\n]+", lambda m: m.group(0).split(":", 1)[0] + ": <redacted:medical_records>"),
        (r"(?i)\b(?:fingerprints|retina/?iris\s+scan|voice\s+signature|facial\s+image|fine\s+location|ethnicity|sexual\s+orientation)\s*:\s*[^\n]+", lambda m: m.group(0).split(":", 1)[0] + ": <redacted:sensitive_pii>"),
    ]
    for pattern, replacement in pii_patterns:
        sanitized = re.sub(pattern, replacement, sanitized)
    return sanitized


def _sanitize_llm_text(text: str) -> str:
    return _redact_pii(_neutralize_prompt_injection(text))


def _extract_pdf_text(file_path: Path) -> str:
    reader = PdfReader(str(file_path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _write_sanitized_upload(file_path: Path) -> Path:
    extracted_text = _extract_pdf_text(file_path)
    sanitized_text = _sanitize_llm_text(extracted_text)
    temp_file = tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", prefix="sanitized_upload_", delete=False, encoding="utf-8"
    )
    with temp_file:
        temp_file.write(sanitized_text)
    return Path(temp_file.name)


async def main():
    file_path = [
        Path("data/YamEnterprises_Monotype Fonts Plan License.US.en 04.0 (BLP).pdf")
    ]
    file_path = [
        Path(
            "data/YamEnterprises_Monotype Fonts Plan License.US.en 04.0 (BLP) reduced.pdf"
        )
    ]

    config_file_name = (
        "/Users/jchevall/Coding/quivr/backend/core/tests/rag_config_workflow.yaml"
    )

    assistant_config = AssistantConfig.from_yaml(config_file_name)
    # megaparse_config = find_nested_key(config, "megaparse_config")
    megaparse_config = assistant_config.ingestion_config.parser_config.megaparse_config
    megaparse_config.llama_parse_api_key = os.getenv("LLAMA_PARSE_API_KEY")

    processor_kwargs = {
        "megaparse_config": megaparse_config,
        "splitter_config": assistant_config.ingestion_config.parser_config.splitter_config,
    }

    sanitized_file_paths = [_write_sanitized_upload(path) for path in file_path]

    # SECURITY NOTICE: Brain is not on the organization's approved model/agent list here.
    # Replace Brain with an organization-approved LLM/agent before production use.
    brain = await Brain.afrom_files(
        name="test_brain",
        file_paths=sanitized_file_paths,
        processor_kwargs=processor_kwargs,
    )

    # # Check brain info
    brain.print_info()

    questions = [
        "What is the contact name for Yam Enterprises?",
        "What is the customer contact detail for Yam Enterprises?",
        "What is the Production Fonts (maximum) for Yam Enterprises?",
        "List the past use font software according to past use term for Yam Enterprises.",
        "How many unique Font Name are there in the Add-On Font Software Section for Yam Enterprises?",
        "What is the maximum number of Production Fonts allowed based on the license usage per term for Yam Enterprises?",
        "What is the number of production fonts licensed by Yam Enterprises? List them one by one.",
        "What is the number of Licensed Monthly Page Views for Yam Enterprises?",
        "What is the monthly licensed impressions (Digital Marketing Communications) for Yam Enterprises?",
        "What is the number of Licensed Applications for Yam Enterprises?",
        "For Yam Enterprises what is the number of applications aggregate Registered users?",
        "What is the number of licensed servers for Yam Enterprises?",
        "When is swap of Production Fonts available in Yam Enterprises?",
        "Who is the primary licensed monotype fonts user for Yam Enterprises?",
        "What is the number of Licensed Commercial Electronic Documents for Yam Enterprises?",
        "How many licensed monotype fonts users can Yam Enterprises have?",
        "How many licensed desktop users can Yam Enterprises have?",
        "Which contract type does Yam Enterprises follow?",
        "What monotype fonts support does Yam Enterprises have?",
        "Which monotype font services onboarding does Yam Enterprises have?",
        "Which Font/User Management does Yam Enterprises have?",
        "What Add-on inventory set did Yam Enterprises pick?",
        "Does Yam Enterprises have Single sign on?",
        "Is there Brand and Licence protection for Yam Enterprises?",
        "Who is the Third Party Payor's contact in Yam Enterprises?",
        "Does Yam Enterprises contract have Company Desktop License?",
        "What is the Number of Swaps Allowed for Yam Enterprises?",
        "When is swap of Production Fonts available in Yam Enterprises?",
    ]

    answers = [
        "Haruko Yamamoto",
        "<redacted:phone>",
        "300 Production Fonts",
        "Helvetica Regular",
        "7",
        "300 Production Fonts",
        "Yam Enterprises has licensed a total of 105 Production Fonts.",
        "35,000,000",
        "2,500,000",
        "60",
        "40",
        "2",
        "Once per quarter",
        "Haruko Yamamoto",
        "0",
        "100",
        "60",
        "License",
        "Premier",
        "Premier",
        "Premier",
        "Plus",
        "Yes",
        "Yes",
        """
        Name: Yami Enterprises

        Contact: Mei Mei

        Address: 20-22 Tsuki-Tsuki-dori, Tokyo, Japan

        Phone: +81 71-9336-54023

        E-mail: mei.mei@example.com
        """,
        "Yes",
        "One (1) swap per calendar quarter",
        "The swap of Production Fonts will be available one (1) time per calendar quarter by removing Font Software as a Production Font and choosing other Font Software on the Monotype Fonts Platform.",
    ]

    retrieval_config = assistant_config.retrieval_config
    for i, (question, truth) in enumerate(zip(questions, answers, strict=False)):
        question = _sanitize_llm_text(question)
        chunk = brain.ask(question=question, retrieval_config=retrieval_config)
        print(
            "\n Question: ", question, "\n Answer: ", chunk.answer, "\n Truth: ", truth
        )
        if i == 5:
            break


if __name__ == "__main__":
    dotenv.load_dotenv()

    # Run the main function in the existing event loop
    asyncio.run(main())
