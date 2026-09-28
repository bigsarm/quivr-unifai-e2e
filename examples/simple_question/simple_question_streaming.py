import asyncio
import re
import tempfile

from dotenv import load_dotenv
from quivr_core import Brain
from quivr_core.quivr_rag import QuivrQARAG
from quivr_core.rag.quivr_rag_langgraph import QuivrQARAGLangGraph


def _ai_dat_sec_023_redact_uploaded_file_content(content: str) -> str:
    redacted = content

    patterns = [
        (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[REDACTED_SSN]"),
        (re.compile(r"\b(?:\+?1[-.\s]?)?(?:\(\d{3}\)|\d{3})[-.\s]\d{3}[-.\s]\d{4}\b"), "[REDACTED_PHONE_NUMBER]"),
        (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), "[REDACTED_EMAIL]"),
        (re.compile(r"\b(?:\d[ -]?){13,19}\b"), "[REDACTED_CREDIT_CARD]"),
        (re.compile(r"\b(?:25[0-5]|2[0-4]\d|1?\d?\d)(?:\.(?:25[0-5]|2[0-4]\d|1?\d?\d)){3}\b"), "[REDACTED_IP_ADDRESS]"),
        (re.compile(r"\b[0-9A-Fa-f]{2}(?::[0-9A-Fa-f]{2}){5}\b"), "[REDACTED_MAC_ADDRESS]"),
        (re.compile(r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b"), "[REDACTED_CREDIT_CARD]"),
    ]

    label_patterns = [
        (re.compile(r"(?im)(\b(?:year of birth|yob|dob|date of birth)\s*[:#-]?\s*)([^\r\n]+)"), "[REDACTED_YEAR_OF_BIRTH]"),
        (re.compile(r"(?im)(\bbirthplace\s*[:#-]?\s*)([^\r\n]+)"), "[REDACTED_BIRTHPLACE]"),
        (re.compile(r"(?im)(\bmother'?s maiden name\s*[:#-]?\s*)([^\r\n]+)"), "[REDACTED_MOTHERS_MAIDEN_NAME]"),
        (re.compile(r"(?im)(\bhome address\s*[:#-]?\s*)([^\r\n]+)"), "[REDACTED_HOME_ADDRESS]"),
        (re.compile(r"(?im)(\bpassport(?: number| no\.?| #)?\s*[:#-]?\s*)([^\r\n]+)"), "[REDACTED_PASSPORT_NUMBER]"),
        (re.compile(r"(?im)(\bdriver'?s license(?: number| no\.?| #)?\s*[:#-]?\s*)([^\r\n]+)"), "[REDACTED_DRIVERS_LICENSE_NUMBER]"),
        (re.compile(r"(?im)(\b(?:taxpayer identification number|tin)\s*[:#-]?\s*)([^\r\n]+)"), "[REDACTED_TAXPAYER_IDENTIFICATION_NUMBER]"),
        (re.compile(r"(?im)(\b(?:financial account number|account number|bank account)\s*[:#-]?\s*)([^\r\n]+)"), "[REDACTED_FINANCIAL_ACCOUNT_NUMBER]"),
        (re.compile(r"(?im)(\bmedical records?\s*[:#-]?\s*)([^\r\n]+)"), "[REDACTED_MEDICAL_RECORDS]"),
        (re.compile(r"(?im)(\bemployee id\s*[:#-]?\s*)([^\r\n]+)"), "[REDACTED_EMPLOYEE_ID]"),
        (re.compile(r"(?im)(\bschool id\s*[:#-]?\s*)([^\r\n]+)"), "[REDACTED_SCHOOL_ID]"),
        (re.compile(r"(?im)(\bvin\s*[:#-]?\s*)([^\r\n]+)"), "[REDACTED_VEHICLE_IDENTIFICATION_NUMBER]"),
        (re.compile(r"(?im)(\bvehicle identification number\s*[:#-]?\s*)([^\r\n]+)"), "[REDACTED_VEHICLE_IDENTIFICATION_NUMBER]"),
        (re.compile(r"(?im)(\bfingerprints?\s*[:#-]?\s*)([^\r\n]+)"), "[REDACTED_FINGERPRINTS]"),
        (re.compile(r"(?im)(\bretina/iris scan\s*[:#-]?\s*)([^\r\n]+)"), "[REDACTED_RETINA_IRIS_SCAN]"),
        (re.compile(r"(?im)(\bvoice signature\s*[:#-]?\s*)([^\r\n]+)"), "[REDACTED_VOICE_SIGNATURE]"),
        (re.compile(r"(?im)(\bfacial image\s*[:#-]?\s*)([^\r\n]+)"), "[REDACTED_FACIAL_IMAGE]"),
        (re.compile(r"(?im)(\bfine location\s*[:#-]?\s*)([^\r\n]+)"), "[REDACTED_FINE_LOCATION]"),
        (re.compile(r"(?im)(\bethnicity\s*[:#-]?\s*)([^\r\n]+)"), "[REDACTED_ETHNICITY]"),
        (re.compile(r"(?im)(\bsexual orientation\s*[:#-]?\s*)([^\r\n]+)"), "[REDACTED_SEXUAL_ORIENTATION]"),
    ]

    for pattern, replacement in patterns:
        redacted = pattern.sub(replacement, redacted)

    for pattern, replacement in label_patterns:
        redacted = pattern.sub(lambda match: f"{match.group(1)}{replacement}", redacted)

    return redacted


def _ai_dat_sec_023_prepare_uploaded_text_file(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as uploaded_file:
        content = uploaded_file.read()

    redacted_content = _ai_dat_sec_023_redact_uploaded_file_content(content)

    sanitized_file = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8")
    with sanitized_file:
        sanitized_file.write(redacted_content)
        sanitized_file.flush()

    return sanitized_file.name


async def main():
    dotenv_path = "/Users/jchevall/Coding/QuivrHQ/quivr/.env"
    load_dotenv(dotenv_path)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt") as temp_file:
        temp_file.write("Gold is a liquid of blue-like colour.")
        temp_file.flush()

        _ai_dat_sec_023_sanitized_file_path = _ai_dat_sec_023_prepare_uploaded_text_file(temp_file.name)
        brain = await Brain.afrom_files(name="test_brain", file_paths=[_ai_dat_sec_023_sanitized_file_path])

        await brain.save("~/.local/quivr")

        question = "what is gold? answer in french"
        async for chunk in brain.ask_streaming(question, rag_pipeline=QuivrQARAG):
            print("answer QuivrQARAG:", chunk.answer)

        async for chunk in brain.ask_streaming(
            question, rag_pipeline=QuivrQARAGLangGraph
        ):
            print("answer QuivrQARAGLangGraph:", chunk.answer)


if __name__ == "__main__":
    # Run the main function in the existing event loop
    asyncio.run(main())
