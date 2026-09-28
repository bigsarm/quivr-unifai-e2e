import tempfile
import re

from quivr_core import Brain

import dotenv

dotenv.load_dotenv()

_ai_dat_sec_023_PATTERNS = [
    ("ssn", re.compile(r"\b\d{3}[- ]\d{2}[- ]\d{4}\b")),
    ("phone", re.compile(r"(?:\+1[ .-]?)?(?:\(\d{3}\)|\b\d{3})[ .-]?\d{3}[ .-]?\d{4}\b")),
    ("email", re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")),
    ("address", re.compile(r"\b\d{1,5}\s+(?:[A-Z][a-z]+\s){1,3}(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|Way)\b\.?(?:,\s*[A-Z][a-z]+(?:\s[A-Z][a-z]+)*)?(?:,\s*[A-Z]{2}\b(?:\s+\d{5}(?:-\d{4})?)?)?(?:,\s*(?:USA|United States)\b)?")),
    ("dob", re.compile(r"(?i)\b(?:DOB|date of birth|born(?: on| in)?)\s*:?\s*(?:\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4}|(?:19|20)\d{2})\b")),
    ("passport", re.compile(r"(?i)\bpassport(?:\s*(?:no\.?|number|#))?\s*:?\s*(?=[A-Z0-9]*\d)[A-Z0-9]{6,9}\b")),
    ("drivers_license", re.compile(r"(?i)\b(?:driver'?s license|drivers license|dl)\s*(?:no\.?|number|#)?\s*:?\s*[A-Z0-9-]{4,20}\b")),
    ("tax_id", re.compile(r"(?i)\b(?:taxpayer identification number|tax id|tin|ein)\s*:?\s*\d{2}-?\d{7}\b")),
    ("credit_card", re.compile(r"\b(?:\d[ -]*?){13,19}\b")),
    ("account_number", re.compile(r"(?i)\b(?:account(?: number| no\.?)?|financial account(?: number)?)\s*:?\s*[A-Z0-9-]{6,20}\b")),
    ("employee_id", re.compile(r"(?i)\bemployee id\s*:?\s*[A-Z0-9-]{2,20}\b")),
    ("school_id", re.compile(r"(?i)\bschool id\s*:?\s*[A-Z0-9-]{2,20}\b")),
    ("vin", re.compile(r"(?i)\b(?:vehicle identification number|vin)\s*:?\s*[A-HJ-NPR-Z0-9]{17}\b")),
    ("ip_address", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
    ("mac_address", re.compile(r"\b(?:[0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}\b|\b(?:[0-9A-Fa-f]{2}-){5}[0-9A-Fa-f]{2}\b")),
    ("birthplace", re.compile(r"(?i)\bbirthplace\s*:?\s*[^\n,;]+")),
    ("maiden_name", re.compile(r"(?i)\b(?:mother'?s maiden name|maiden name)\s*:?\s*[^\n,;]+")),
    ("medical", re.compile(r"(?i)\b(?:medical records?|medical record)\s*:?\s*[^\n]+")),
    ("location", re.compile(r"(?i)\b(?:fine location|location)\s*:?\s*[^\n,;]+")),
    ("ethnicity", re.compile(r"(?i)\bethnicity\s*:?\s*[^\n,;]+")),
    ("sexual_orientation", re.compile(r"(?i)\bsexual orientation\s*:?\s*[^\n,;]+")),
]


def _ai_dat_sec_023_redact_uploaded_text(text):
    for _ai_dat_sec_023_category, _ai_dat_sec_023_pattern in _ai_dat_sec_023_PATTERNS:
        text = _ai_dat_sec_023_pattern.sub(f"<redacted:{_ai_dat_sec_023_category}>", text)
    return text

if __name__ == "__main__":
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt") as temp_file:
        temp_file.write("Gold is a liquid of blue-like colour.")
        temp_file.flush()

        with open(temp_file.name, "r", encoding="utf-8") as _ai_dat_sec_023_uploaded_file:
            _ai_dat_sec_023_redacted_text = _ai_dat_sec_023_redact_uploaded_text(_ai_dat_sec_023_uploaded_file.read())

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", encoding="utf-8") as _ai_dat_sec_023_redacted_file:
            _ai_dat_sec_023_redacted_file.write(_ai_dat_sec_023_redacted_text)
            _ai_dat_sec_023_redacted_file.flush()

            brain = Brain.from_files(
                name="test_brain",
                file_paths=[_ai_dat_sec_023_redacted_file.name],
            )

        answer = brain.ask("what is gold? answer in french")
        print("answer QuivrQARAGLangGraph :", answer)
