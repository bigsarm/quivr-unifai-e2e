import asyncio
import logging
import os
import re
import tempfile
from pathlib import Path

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


_ai_dat_sec_023_PII_PATTERNS = {
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "year_of_birth": re.compile(r"\b(?:19\d{2}|20(?:0\d|1\d|2[0-4]))\b"),
    "birthplace": re.compile(r"\bBirthplace\s*:\s*[^\n\r]+", re.IGNORECASE),
    "personal_phone": re.compile(r"\b(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{2,4}\)?[\s.-]?)\d{3,4}[\s.-]?\d{4,}\b"),
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "mothers_maiden_name": re.compile(r"\bMother'?s Maiden Name\s*:\s*[^\n\r]+", re.IGNORECASE),
    "home_address": re.compile(r"\bAddress\s*:\s*[^\n\r]+", re.IGNORECASE),
    "passport_number": re.compile(r"\b[A-Z0-9]{6,9}\b"),
    "drivers_license_number": re.compile(r"\bDriver'?s License Number\s*:\s*[^\n\r]+", re.IGNORECASE),
    "tin": re.compile(r"\b\d{2}-\d{7}\b"),
    "credit_card": re.compile(r"\b(?:\d[ -]*?){13,19}\b"),
    "financial_account": re.compile(r"\bAccount Number\s*:\s*[^\n\r]+", re.IGNORECASE),
    "employee_id": re.compile(r"\bEmployee Id\s*:\s*[^\n\r]+", re.IGNORECASE),
    "school_id": re.compile(r"\bSchool Id\s*:\s*[^\n\r]+", re.IGNORECASE),
    "vin": re.compile(r"\b[A-HJ-NPR-Z0-9]{17}\b"),
    "ip_address": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    "mac_address": re.compile(r"\b(?:[0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}\b"),
    "fine_location": re.compile(r"\b(?:Latitude|Longitude|Coordinates)\s*:\s*[^\n\r]+", re.IGNORECASE),
    "ethnicity": re.compile(r"\bEthnicity\s*:\s*[^\n\r]+", re.IGNORECASE),
    "sexual_orientation": re.compile(r"\bSexual Orientation\s*:\s*[^\n\r]+", re.IGNORECASE),
}


def _ai_dat_sec_023_redact_text(_ai_dat_sec_023_text: str) -> str:
    _ai_dat_sec_023_redacted = _ai_dat_sec_023_text
    for _ai_dat_sec_023_label, _ai_dat_sec_023_pattern in _ai_dat_sec_023_PII_PATTERNS.items():
        _ai_dat_sec_023_redacted = _ai_dat_sec_023_pattern.sub(
            f"[REDACTED {_ai_dat_sec_023_label.upper()}]", _ai_dat_sec_023_redacted
        )
    return _ai_dat_sec_023_redacted


def _ai_dat_sec_023_prepare_uploaded_file(_ai_dat_sec_023_path: Path) -> Path:
    _ai_dat_sec_023_bytes = _ai_dat_sec_023_path.read_bytes()
    _ai_dat_sec_023_text = _ai_dat_sec_023_bytes.decode("latin-1")
    _ai_dat_sec_023_redacted_text = _ai_dat_sec_023_redact_text(_ai_dat_sec_023_text)
    if _ai_dat_sec_023_redacted_text == _ai_dat_sec_023_text:
        return _ai_dat_sec_023_path

    _ai_dat_sec_023_suffix = _ai_dat_sec_023_path.suffix or ".tmp"
    with tempfile.NamedTemporaryFile(delete=False, suffix=_ai_dat_sec_023_suffix) as _ai_dat_sec_023_tmp:
        _ai_dat_sec_023_tmp.write(_ai_dat_sec_023_redacted_text.encode("latin-1"))
        return Path(_ai_dat_sec_023_tmp.name)


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

    file_path = [_ai_dat_sec_023_prepare_uploaded_file(path) for path in file_path]

    brain = await Brain.afrom_files(
        name="test_brain",
        file_paths=file_path,
        processor_kwargs=processor_kwargs,
    )

    # # Check brain info
    brain.print_info()

    questions = [
        "What is the contact name for Yam Enterprises?",
        "What is the customer phone for Yam Enterprises?",
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
        "81 90-1234-5603",
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
