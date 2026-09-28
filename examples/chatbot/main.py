import tempfile
import base64
import binascii
import codecs
import re
import urllib.parse

import chainlit as cl
from quivr_core import Brain
from quivr_core.rag.entities.config import RetrievalConfig

_PII_PATTERNS = [
    ("ssn", re.compile(r"\b\d{3}[- ]\d{2}[- ]\d{4}\b")),
    ("phone", re.compile(r"(?:\+1[ .-]?)?(?:\(\d{3}\)|\b\d{3})[ .-]?\d{3}[ .-]?\d{4}\b")),
    ("email", re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")),
    ("address", re.compile(r"\b\d{1,5}\s+(?:[A-Z][a-z]+\s){1,3}(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|Way)\b\.?(?:,\s*[A-Z][a-z]+(?:\s[A-Z][a-z]+)*)?(?:,\s*[A-Z]{2}\b(?:\s+\d{5}(?:-\d{4})?)?)?(?:,\s*(?:USA|United States)\b)?")),
    ("dob", re.compile(r"(?i)\b(?:DOB|date of birth|born(?: on| in)?)\s*:?\s*(\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4}|(?:19|20)\d{2})\b")),
    ("passport", re.compile(r"(?i)\b(passport(?:\s*(?:no\.?|number|#))?\s*:?\s*)(?=[A-Z0-9]*\d)([A-Z0-9]{6,9})\b")),
    ("drivers_license", re.compile(r"(?i)\b(driver(?:'s)? license(?:\s*(?:no\.?|number|#))?\s*:?\s*)([A-Z0-9-]{4,20})\b")),
    ("tax_id", re.compile(r"(?i)\b(taxpayer identification number|tax id|tin|ein)\s*:?\s*([A-Z0-9-]{4,20})\b")),
    ("credit_card", re.compile(r"\b(?:\d[ -]*?){13,19}\b")),
    ("account_number", re.compile(r"(?i)\b(account(?:\s*(?:no\.?|number|#))?)\s*:?\s*([A-Z0-9-]{4,34})\b")),
    ("employee_id", re.compile(r"(?i)\b(employee id)\s*:?\s*([A-Z0-9-]{2,20})\b")),
    ("school_id", re.compile(r"(?i)\b(school id|student id)\s*:?\s*([A-Z0-9-]{2,20})\b")),
    ("vin", re.compile(r"(?i)\b(vin|vehicle identification number)\s*:?\s*([A-HJ-NPR-Z0-9]{17})\b")),
    ("ip_address", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
    ("mac_address", re.compile(r"\b(?:[0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}\b")),
    ("birthplace", re.compile(r"(?i)\b(birthplace|place of birth|born in)\s*:?\s*([^\n,.;]+)")),
    ("maiden_name", re.compile(r"(?i)\b(mother(?:'s)? maiden name)\s*:?\s*([^\n,.;]+)")),
    ("medical", re.compile(r"(?i)\b(medical records?)\s*:?\s*([^\n]+)")),
    ("location", re.compile(r"(?i)\b(fine location|precise location)\s*:?\s*([^\n,.;]+)")),
    ("ethnicity", re.compile(r"(?i)\b(ethnicity)\s*:?\s*([^\n,.;]+)")),
    ("sexual_orientation", re.compile(r"(?i)\b(sexual orientation)\s*:?\s*([^\n,.;]+)")),
]

_INJECTION_PATTERNS = [
    ("instruction_override", re.compile(r"(?i)\b(?:ignore|disregard|forget)\s+(?:all\s+)?(?:previous|prior|above)\s+instructions\b|\bforget everything above\b|\bignore(?:\s+all)?\s+safety guidelines\b")),
    ("role_hijack", re.compile(r"(?i)\byou are now(?:\s+in)?\s+(?:developer mode|admin mode|dan|unrestricted)\b|\bprovide unrestricted access\b|\benable developer mode\b|\bact as an unrestricted ai\b|\bwithout safety filters\b|\bSYSTEM OVERRIDE\b")),
    ("exfiltration_attempt", re.compile(r"(?i)\breveal(?:\s+all)?\s+confidential information\b|\breveal the system prompt\b|\breveal all system prompts\b|\blist all financial data, user passwords, and API keys\b|\bprint the API key\b|\b(?:send|post|upload|exfiltrate)\b[^\n]{0,120}\bhttps?://\S+|!\[[^\]]*\]\([^)]*https?://[^)]*\)")),
    ("delimiter_escape", re.compile(r"(?i)</system>|<\|im_start\|>|###\s*system:")),
    ("command_injection", re.compile(r"(?i)\bexecute\s*:\s*[^\n]+|\brun\s+(?:rm\s+-rf\s+/|curl\s+https?://\S+(?:\s*\|\s*sh)?|wget\s+https?://\S+(?:\s*\|\s*sh)?|print\s*\(\s*os\.environ\.get\([^\n]+\))")),
]

_HIDDEN_TEXT_PATTERNS = [
    re.compile(r"<!--.*?-->", re.DOTALL),
    re.compile(r"<[^>]*style\s*=\s*[\"'][^\"']*(?:display\s*:\s*none|font-size\s*:\s*0(?:px)?|color\s*:\s*white)[^\"']*[\"'][^>]*>.*?</[^>]+>", re.IGNORECASE | re.DOTALL),
    re.compile(r"\u200b+"),
]


def _replace_group(match: re.Match, category: str, value_group: int, mode: str) -> str:
    value = match.group(value_group)
    if mode == "mask":
        if category in {"ssn", "credit_card", "account_number"}:
            digits = re.sub(r"\D", "", value)
            if len(digits) >= 4:
                return match.group(0).replace(value, f"***{digits[-4:]}")
        replacement = f"<masked:{category}>"
    else:
        replacement = f"<redacted:{category}>"
    return match.group(0).replace(value, replacement)


def _apply_pii_transform(text: str, mode: str) -> str:
    if not text:
        return text
    transformed = text
    for category, pattern in _PII_PATTERNS:
        if category in {"dob"}:
            transformed = pattern.sub(lambda m: _replace_group(m, category, 1, mode), transformed)
        elif category in {"passport", "drivers_license", "tax_id", "account_number", "employee_id", "school_id", "vin"}:
            transformed = pattern.sub(lambda m: _replace_group(m, category, 2, mode), transformed)
        elif category in {"birthplace", "maiden_name", "medical", "location", "ethnicity", "sexual_orientation"}:
            transformed = pattern.sub(lambda m: _replace_group(m, category, 2, mode), transformed)
        elif category in {"ssn", "phone", "email", "address", "credit_card", "ip_address", "mac_address"}:
            replacement = f"<masked:{category}>" if mode == "mask" else f"<redacted:{category}>"
            if mode == "mask" and category in {"ssn", "credit_card"}:
                transformed = pattern.sub(lambda m: _replace_group(m, category, 0, mode), transformed)
            else:
                transformed = pattern.sub(replacement, transformed)
    return transformed


def redact_pii(text: str) -> str:
    return _apply_pii_transform(text, "redact")


def mask_pii(text: str) -> str:
    return _apply_pii_transform(text, "mask")


def _normalized_text_with_map(text: str) -> tuple[str, list[int]]:
    normalized_chars = []
    index_map = []
    substitutions = {"1": "i", "3": "e", "0": "o", "4": "a", "5": "s", "7": "t"}
    for index, char in enumerate(text):
        mapped = substitutions.get(char.lower(), char.lower())
        if mapped.isspace():
            continue
        normalized_chars.append(mapped)
        index_map.append(index)
    return "".join(normalized_chars), index_map


def _replace_span(text: str, start: int, end: int, replacement: str) -> str:
    return text[:start] + replacement + text[end:]


def sanitize_untrusted_text(text: str) -> str:
    if not text:
        return text
    sanitized = text
    for pattern in _HIDDEN_TEXT_PATTERNS:
        sanitized = pattern.sub("<prompt_injection_removed: hidden_text>", sanitized)
    for category, pattern in _INJECTION_PATTERNS:
        sanitized = pattern.sub(f"<prompt_injection_removed: {category}>", sanitized)

    normalized, index_map = _normalized_text_with_map(sanitized)
    leetspeak_patterns = [
        ("instruction_override", re.compile(r"(?:ignore|disregard|forget)(?:all)?(?:previous|prior|above)instructions|forgeteverythingabove|ignore(?:all)?safetyguidelines")),
        ("role_hijack", re.compile(r"youarenow(?:in)?(?:developermode|adminmode|dan|unrestricted)|provideunrestrictedaccess|enabledevelopermode|actasanunrestrictedai|withoutsafetyfilters|systemoverride")),
    ]
    for category, pattern in leetspeak_patterns:
        match = pattern.search(normalized)
        if match:
            start = index_map[match.start()]
            end = index_map[match.end() - 1] + 1
            sanitized = _replace_span(sanitized, start, end, f"<prompt_injection_removed: {category}>")
            normalized, index_map = _normalized_text_with_map(sanitized)

    encoded_spans = []
    for candidate in re.finditer(r"\b[A-Za-z0-9+/=]{12,}\b|\b(?:[0-9A-Fa-f]{2}){6,}\b|%(?:[0-9A-Fa-f]{2}){4,}", sanitized):
        token = candidate.group(0)
        decoded_values = []
        try:
            if re.fullmatch(r"[A-Za-z0-9+/=]{12,}", token):
                decoded_values.append(base64.b64decode(token, validate=True).decode("utf-8", errors="ignore"))
        except (binascii.Error, ValueError):
            pass
        try:
            if re.fullmatch(r"(?:[0-9A-Fa-f]{2}){6,}", token):
                decoded_values.append(bytes.fromhex(token).decode("utf-8", errors="ignore"))
        except ValueError:
            pass
        if "%" in token:
            decoded_values.append(urllib.parse.unquote(token))
        decoded_values.append(codecs.decode(token, "rot13"))
        for decoded in decoded_values:
            for category, pattern in _INJECTION_PATTERNS:
                if pattern.search(decoded):
                    encoded_spans.append((candidate.start(), candidate.end(), f"<prompt_injection_removed: encoded_payload>"))
                    break
            else:
                continue
            break
    for start, end, replacement in reversed(encoded_spans):
        sanitized = _replace_span(sanitized, start, end, replacement)

    return sanitized


@cl.on_chat_start
async def on_chat_start():
    files = None

    # Wait for the user to upload a file
    while files is None:
        files = await cl.AskFileMessage(
            content="Please upload a text .txt file to begin!",
            accept=["text/plain"],
            max_size_mb=20,
            timeout=180,
        ).send()

    file = files[0]

    msg = cl.Message(content=f"Processing `{file.name}`...")
    await msg.send()

    with open(file.path, "r", encoding="utf-8") as f:
        text = f.read()

    text = sanitize_untrusted_text(text)
    text = redact_pii(text)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=file.name, delete=False
    ) as temp_file:
        temp_file.write(text)
        temp_file.flush()
        temp_file_path = temp_file.name

    brain = Brain.from_files(name="user_brain", file_paths=[temp_file_path])

    # Store the file path in the session
    cl.user_session.set("file_path", temp_file_path)

    # Let the user know that the system is ready
    msg.content = f"Processing `{file.name}` done. You can now ask questions!"
    await msg.update()

    cl.user_session.set("brain", brain)


@cl.on_message
async def main(message: cl.Message):
    brain = cl.user_session.get("brain")  # type: Brain
    path_config = "basic_rag_workflow.yaml"
    retrieval_config = RetrievalConfig.from_yaml(path_config)

    if brain is None:
        await cl.Message(content="Please upload a file first.").send()
        return

    # Prepare the message for streaming
    msg = cl.Message(content="", elements=[])
    await msg.send()

    saved_sources = set()
    saved_sources_complete = []
    elements = []

    # Use the ask_stream method for streaming responses
    sanitized_message = sanitize_untrusted_text(message.content)
    sanitized_message = redact_pii(sanitized_message)
    response_chunks = []
    async for chunk in brain.ask_streaming(sanitized_message, retrieval_config=retrieval_config):
        response_chunks.append(chunk.answer)
        for source in chunk.metadata.sources:
            sanitized_source_content = sanitize_untrusted_text(source.page_content)
            sanitized_source_content = redact_pii(sanitized_source_content)
            if sanitized_source_content not in saved_sources:
                saved_sources.add(sanitized_source_content)
                saved_sources_complete.append(source)
                print(source)
                elements.append(cl.Text(name=source.metadata["original_file_name"], content=mask_pii(sanitized_source_content), display="side"))

    
    msg.content = mask_pii("".join(response_chunks))
    await msg.send()
    sources = ""
    for source in saved_sources_complete:
        sources += f"- {source.metadata['original_file_name']}\n"
    msg.elements = elements
    msg.content = msg.content + f"\n\nSources:\n{sources}"
    await msg.update()