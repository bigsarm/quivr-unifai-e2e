import tempfile
import os
import re

from quivr_core import Brain

import dotenv

dotenv.load_dotenv()


def _redact_uploaded_file_pii(text: str) -> str:
    redacted = text
    patterns = [
        (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[REDACTED_SSN]"),
        (re.compile(r"\b(?:\+?1[\s.-]?)?(?:\(\d{3}\)[\s.-]?|\d{3}[\s.-])\d{3}[\s.-]\d{4}\b"), "[REDACTED_PHONE]"),
        (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), "[REDACTED_EMAIL]"),
        (re.compile(r"\b(?:\d[ -]?){13,19}\b"), "[REDACTED_CARD]"),
        (re.compile(r"\b(?:\d[ -]?){9,17}\b"), "[REDACTED_ACCOUNT]"),
        (re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), "[REDACTED_IP]"),
        (re.compile(r"\b(?:[0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}\b"), "[REDACTED_MAC]"),
        (re.compile(r"\b(?:[A-HJ-NPR-Z0-9]{17})\b"), "[REDACTED_VIN]"),
    ]
    for pattern, marker in patterns:
        redacted = pattern.sub(marker, redacted)

    labeled_patterns = [
        (re.compile(r"\b(Year of Birth\s*:\s*)([^\n\r]+)", re.IGNORECASE), "[REDACTED_YEAR_OF_BIRTH]"),
        (re.compile(r"\b(DOB\s*:\s*)([^\n\r]+)", re.IGNORECASE), "[REDACTED_YEAR_OF_BIRTH]"),
        (re.compile(r"\b(Birthplace\s*:\s*)([^\n\r]+)", re.IGNORECASE), "[REDACTED_BIRTHPLACE]"),
        (re.compile(r"\b(Mother(?:'s)? Maiden Name\s*:\s*)([^\n\r]+)", re.IGNORECASE), "[REDACTED_MAIDEN_NAME]"),
        (re.compile(r"\b(Home Address\s*:\s*)([^\n\r]+)", re.IGNORECASE), "[REDACTED_HOME_ADDRESS]"),
        (re.compile(r"\b(Passport Number\s*:\s*)([^\n\r]+)", re.IGNORECASE), "[REDACTED_PASSPORT]"),
        (re.compile(r"\b(Drivers? License Number\s*:\s*)([^\n\r]+)", re.IGNORECASE), "[REDACTED_DRIVERS_LICENSE]"),
        (re.compile(r"\b(Taxpayer Identification Number\s*:\s*)([^\n\r]+)", re.IGNORECASE), "[REDACTED_TIN]"),
        (re.compile(r"\b(Financial Account Number\s*:\s*)([^\n\r]+)", re.IGNORECASE), "[REDACTED_ACCOUNT]"),
        (re.compile(r"\b(Fingerprints\s*:\s*)([^\n\r]+)", re.IGNORECASE), "[REDACTED_FINGERPRINTS]"),
        (re.compile(r"\b(Retina/Iris Scan\s*:\s*)([^\n\r]+)", re.IGNORECASE), "[REDACTED_RETINA_IRIS]"),
        (re.compile(r"\b(Voice signature\s*:\s*)([^\n\r]+)", re.IGNORECASE), "[REDACTED_VOICE_SIGNATURE]"),
        (re.compile(r"\b(Facial image\s*:\s*)([^\n\r]+)", re.IGNORECASE), "[REDACTED_FACIAL_IMAGE]"),
        (re.compile(r"\b(Medical Records\s*:\s*)([^\n\r]+)", re.IGNORECASE), "[REDACTED_MEDICAL_RECORDS]"),
        (re.compile(r"\b(Employee Id\s*:\s*)([^\n\r]+)", re.IGNORECASE), "[REDACTED_EMPLOYEE_ID]"),
        (re.compile(r"\b(School Id\s*:\s*)([^\n\r]+)", re.IGNORECASE), "[REDACTED_SCHOOL_ID]"),
        (re.compile(r"\b(Fine Location\s*:\s*)([^\n\r]+)", re.IGNORECASE), "[REDACTED_FINE_LOCATION]"),
        (re.compile(r"\b(Ethnicity\s*:\s*)([^\n\r]+)", re.IGNORECASE), "[REDACTED_ETHNICITY]"),
        (re.compile(r"\b(Sexual Orientation\s*:\s*)([^\n\r]+)", re.IGNORECASE), "[REDACTED_SEXUAL_ORIENTATION]"),
    ]
    for pattern, marker in labeled_patterns:
        redacted = pattern.sub(lambda match: f"{match.group(1)}{marker}", redacted)

    return redacted


def _prepare_uploaded_file(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as uploaded_file:
        content = uploaded_file.read()

    sanitized_content = _redact_uploaded_file_pii(content)
    sanitized_file = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8")
    try:
        sanitized_file.write(sanitized_content)
        sanitized_file.flush()
    finally:
        sanitized_file.close()
    return sanitized_file.name

if __name__ == "__main__":
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt") as temp_file:
        temp_file.write("Gold is a liquid of blue-like colour.")
        temp_file.flush()

        sanitized_file_path = _prepare_uploaded_file(temp_file.name)
        try:
            brain = Brain.from_files(
                name="test_brain",
                file_paths=[sanitized_file_path],
            )

            answer = brain.ask("what is gold? answer in french")
            print("answer QuivrQARAGLangGraph :", answer)
        finally:
            if os.path.exists(sanitized_file_path):
                os.unlink(sanitized_file_path)

        # Brain usage remains unchanged here; replace it with an organization-approved LLM outside this file if registry review determines this agent resolves to a disapproved or otherwise unapproved model.
