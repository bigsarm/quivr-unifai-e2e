import asyncio
import logging
import os
import re
import tempfile
from pathlib import Path

from pypdf import PdfReader, PdfWriter

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


_PHONE_PATTERN = re.compile(r"(?:\+1[ .-]?)?(?:\(\d{3}\)|\b\d{3})[ .-]?\d{3}[ .-]?\d{4}\b|\+\d{1,3}[ .-]?\d{1,4}[ .-]?\d{3,4}[ .-]?\d{4,}")
_EMAIL_PATTERN = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
_SSN_PATTERN = re.compile(r"\b\d{3}[- ]\d{2}[- ]\d{4}\b")
_ADDRESS_PATTERN = re.compile(r"\b\d{1,5}\s+(?:[A-Z][a-z]+(?:[-\s][A-Z][a-z]+){0,2}\s){0,3}(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|Way)\b\.?(?:,\s*[A-Z][a-z]+(?:\s[A-Z][a-z]+)*)?(?:,\s*[A-Z]{2}\b(?:\s+\d{5}(?:-\d{4})?)?)?(?:,\s*(?:USA|United States)\b)?")
_CREDIT_CARD_PATTERN = re.compile(r"\b(?:\d[ -]*?){13,19}\b")
_IP_ADDRESS_PATTERN = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_MAC_ADDRESS_PATTERN = re.compile(r"\b(?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}\b")
_DOB_PATTERN = re.compile(r"(?i)\b(?:DOB|date of birth|born(?: on| in)?)\s*:?\s*(?:\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4}|(?:19|20)\d{2})\b")
_PASSPORT_PATTERN = re.compile(r"(?i)\bpassport(?:\s*(?:no\.?|number|#))?\s*:?\s*(?=[A-Z0-9]*\d)[A-Z0-9]{6,9}\b")
_DRIVERS_LICENSE_PATTERN = re.compile(r"(?i)\b(?:driver'?s license|drivers license|driver license)(?:\s*(?:no\.?|number|#))?\s*:?\s*[A-Z0-9-]{4,20}\b")
_TAX_ID_PATTERN = re.compile(r"(?i)\b(?:taxpayer identification number|tax id|tin)(?:\s*(?:no\.?|number|#))?\s*:?\s*[A-Z0-9-]{6,20}\b")
_ACCOUNT_NUMBER_PATTERN = re.compile(r"(?i)\b(?:financial account|account number|account no\.?)(?:\s*(?:number|#))?\s*:?\s*[A-Z0-9-]{6,20}\b")
_EMPLOYEE_ID_PATTERN = re.compile(r"(?i)\bemployee id\s*:?\s*[A-Z0-9-]{2,20}\b")
_SCHOOL_ID_PATTERN = re.compile(r"(?i)\bschool id\s*:?\s*[A-Z0-9-]{2,20}\b")
_VIN_PATTERN = re.compile(r"(?i)\b(?:vehicle identification number|vin)\s*:?\s*[A-HJ-NPR-Z0-9]{11,17}\b")
_BIRTHPLACE_PATTERN = re.compile(r"(?i)\b(?:birthplace|place of birth|born in)\s*:?\s*[^\n,;]+")
_MAIDEN_NAME_PATTERN = re.compile(r"(?i)\b(?:mother'?s maiden name|maiden name)\s*:?\s*[^\n,;]+")
_MEDICAL_PATTERN = re.compile(r"(?i)\bmedical records?\s*:?\s*[^\n]+")
_LOCATION_PATTERN = re.compile(r"(?i)\b(?:fine location|precise location|exact location|gps coordinates?)\s*:?\s*[^\n,;]+")
_ETHNICITY_PATTERN = re.compile(r"(?i)\bethnicity\s*:?\s*[^\n,;]+")
_SEXUAL_ORIENTATION_PATTERN = re.compile(r"(?i)\bsexual orientation\s*:?\s*[^\n,;]+")

_REDACTION_PATTERNS = [
    ("ssn", _SSN_PATTERN),
    ("phone", _PHONE_PATTERN),
    ("email", _EMAIL_PATTERN),
    ("address", _ADDRESS_PATTERN),
    ("dob", _DOB_PATTERN),
    ("passport", _PASSPORT_PATTERN),
    ("drivers_license", _DRIVERS_LICENSE_PATTERN),
    ("tax_id", _TAX_ID_PATTERN),
    ("credit_card", _CREDIT_CARD_PATTERN),
    ("account_number", _ACCOUNT_NUMBER_PATTERN),
    ("employee_id", _EMPLOYEE_ID_PATTERN),
    ("school_id", _SCHOOL_ID_PATTERN),
    ("vin", _VIN_PATTERN),
    ("ip_address", _IP_ADDRESS_PATTERN),
    ("mac_address", _MAC_ADDRESS_PATTERN),
    ("birthplace", _BIRTHPLACE_PATTERN),
    ("maiden_name", _MAIDEN_NAME_PATTERN),
    ("medical", _MEDICAL_PATTERN),
    ("location", _LOCATION_PATTERN),
    ("ethnicity", _ETHNICITY_PATTERN),
    ("sexual_orientation", _SEXUAL_ORIENTATION_PATTERN),
]

_LAST4_CATEGORIES = {"ssn", "credit_card", "account_number"}


def redact_pii(text: str) -> str:
    if not isinstance(text, str):
        return text
    redacted = text
    for category, pattern in _REDACTION_PATTERNS:
        redacted = pattern.sub(f"<redacted:{category}>", redacted)
    return redacted



def mask_pii(text: str) -> str:
    if not isinstance(text, str):
        return text
    masked = text
    for category, pattern in _REDACTION_PATTERNS:
        if category in _LAST4_CATEGORIES:
            def _mask_last4(match, pii_category=category):
                value = match.group(0)
                digits = re.sub(r"\D", "", value)
                if len(digits) >= 4:
                    return f"<masked:{pii_category}:{digits[-4:]}>"
                return f"<masked:{pii_category}>"

            masked = pattern.sub(_mask_last4, masked)
        else:
            masked = pattern.sub(f"<masked:{category}>", masked)
    return masked



def _write_redacted_pdf_copy(source_path: Path) -> str:
    reader = PdfReader(str(source_path))
    writer = PdfWriter()
    extracted_chunks = []
    for page in reader.pages:
        extracted_chunks.append(page.extract_text() or "")
        writer.add_page(page)
    writer.add_metadata({"/RedactedText": redact_pii("\n".join(extracted_chunks))})
    temp_file = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    with open(temp_file.name, "wb") as redacted_pdf:
        writer.write(redacted_pdf)
    return temp_file.name


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

    sanitized_file_paths = [
        Path(_write_redacted_pdf_copy(path)) for path in file_path
    ]

    brain = await Brain.afrom_files(
        name="test_brain",
        file_paths=sanitized_file_paths,
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
        question = redact_pii(question)
        chunk = brain.ask(question=question, retrieval_config=retrieval_config)
        answer_text = redact_pii(chunk.answer)
        print(
            "\n Question: ",
            mask_pii(question),
            "\n Answer: ",
            mask_pii(answer_text),
            "\n Truth: ",
            mask_pii(truth),
        )
        if i == 5:
            break


if __name__ == "__main__":
    dotenv.load_dotenv()

    # Run the main function in the existing event loop
    asyncio.run(main())
