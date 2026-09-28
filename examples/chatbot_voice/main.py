import tempfile
import os
import re
import base64
import binascii
import urllib.parse
import codecs
import chainlit as cl
from quivr_core import Brain
from quivr_core.rag.entities.config import RetrievalConfig
from openai import AsyncOpenAI
from chainlit.element import Element

from io import BytesIO


_PII_PATTERNS = [
    ("ssn", re.compile(r"\b\d{3}[- ]\d{2}[- ]\d{4}\b")),
    ("phone", re.compile(r"(?:\+1[ .-]?)?(?:\(\d{3}\)|\b\d{3})[ .-]?\d{3}[ .-]?\d{4}\b")),
    ("email", re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")),
    ("address", re.compile(r"\b\d{1,5}\s+(?:[A-Z][a-z]+\s){1,3}(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|Way)\b\.?(?:,\s*[A-Z][a-z]+(?:\s[A-Z][a-z]+)*)?(?:,\s*[A-Z]{2}\b(?:\s+\d{5}(?:-\d{4})?)?)?(?:,\s*(?:USA|United States)\b)?")),
    ("dob", re.compile(r"(?i)\b(?:DOB|date of birth|born(?: on| in)?)\s*:?\s*(?:\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4}|(?:19|20)\d{2})\b")),
    ("passport", re.compile(r"(?i)\bpassport(?:\s*(?:no\.?|number|#))?\s*:?\s*(?=[A-Z0-9]*\d)[A-Z0-9]{6,9}\b")),
    ("drivers_license", re.compile(r"(?i)\b(?:driver'?s license|drivers license|dl)(?:\s*(?:no\.?|number|#))?\s*:?\s*[A-Z0-9-]{4,20}\b")),
    ("tax_id", re.compile(r"(?i)\b(?:taxpayer identification number|tax id|tin|itin|ein)(?:\s*(?:no\.?|number|#))?\s*:?\s*[A-Z0-9-]{6,20}\b")),
    ("credit_card", re.compile(r"\b(?:\d[ -]*?){13,19}\b")),
    ("account_number", re.compile(r"(?i)\b(?:account number|acct(?:ount)?(?:\s*no\.?)?)(?:\s*(?:number|#|no\.?))?\s*:?\s*[A-Z0-9-]{6,20}\b")),
    ("employee_id", re.compile(r"(?i)\bemployee id\s*:?\s*[A-Z0-9-]{2,20}\b")),
    ("school_id", re.compile(r"(?i)\bschool id\s*:?\s*[A-Z0-9-]{2,20}\b")),
    ("vin", re.compile(r"(?i)\bvin\s*:?\s*[A-HJ-NPR-Z0-9]{17}\b")),
    ("ip_address", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
    ("mac_address", re.compile(r"\b(?:[0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}\b")),
    ("birthplace", re.compile(r"(?i)\bbirthplace\s*:?\s*[^\n,;]+")),
    ("maiden_name", re.compile(r"(?i)\bmother'?s maiden name\s*:?\s*[^\n,;]+")),
    ("medical", re.compile(r"(?i)\bmedical records?\s*:?\s*[^\n]+")),
    ("location", re.compile(r"(?i)\b(?:fine location|location)\s*:?\s*[^\n,;]+")),
    ("ethnicity", re.compile(r"(?i)\bethnicity\s*:?\s*[^\n,;]+")),
    ("sexual_orientation", re.compile(r"(?i)\bsexual orientation\s*:?\s*[^\n,;]+")),
]


def _mask_identifier_value(value: str) -> str:
    digits = re.sub(r"\D", "", value)
    if len(digits) >= 4:
        return f"***{digits[-4:]}"
    return "***"


_IDENTIFIER_CATEGORIES = {"ssn", "credit_card", "account_number"}


def redact_pii(text: str) -> str:
    if not text:
        return text
    redacted = text
    for category, pattern in _PII_PATTERNS:
        redacted = pattern.sub(f"<redacted:{category}>", redacted)
    return redacted


def mask_pii(text: str) -> str:
    if not text:
        return text
    masked = text
    for category, pattern in _PII_PATTERNS:
        if category in _IDENTIFIER_CATEGORIES:
            masked = pattern.sub(lambda match, c=category: _mask_identifier_value(match.group(0)), masked)
        else:
            masked = pattern.sub(f"<masked:{category}>", masked)
    return masked


def _contains_prompt_attack(text: str) -> str | None:
    lowered = text.lower()
    instruction_override = [
        r"\b(?:ignore|disregard|forget)\s+(?:all\s+)?(?:previous|prior|above)\s+instructions\b",
        r"\bforget\s+everything\s+above\b",
        r"\bignore\s+(?:all\s+)?safety\s+guidelines\b",
    ]
    role_hijack = [
        r"\byou\s+are\s+now\s+(?:in\s+)?developer\s+mode\b",
        r"\byou\s+are\s+now\s+in\s+admin\s+mode\b",
        r"\byou\s+are\s+now\s+dan\b",
        r"\byou\s+are\s+now\s+unrestricted\b",
        r"\bprovide\s+unrestricted\s+access\b",
        r"\benable\s+developer\s+mode\b",
        r"\bact\s+as\s+an\s+unrestricted\s+ai\b",
        r"\bwithout\s+safety\s+filters\b",
        r"\bsystem\s+override\b",
    ]
    exfiltration = [
        r"\breveal\s+(?:all\s+)?confidential\s+information\b",
        r"\breveal\s+(?:all\s+)?system\s+prompts?\b",
        r"\blist\s+all\s+financial\s+data,\s*user\s+passwords,\s*and\s+api\s+keys\b",
        r"\bprint\s+the\s+api\s+key\b",
        r"\b(?:send|post|upload|exfiltrate)\b.{0,80}\bhttps?://\S+",
        r"!\[[^\]]*\]\(https?://[^)]+\)",
    ]
    delimiter_escape = [
        r"</system>",
        r"<\|im_start\|>",
        r"###\s*system:",
    ]
    command_injection = [
        r"\bexecute\s*:\s*[^\n]+",
        r"\brun\s+(?:rm\s+-rf\s+/|curl\s+https?://\S+(?:\s*\|\s*sh)?)\b",
        r"\bcurl\s+https?://\S+(?:\s*\|\s*sh)?\b",
        r"\b(?:python|bash|sh)\s+-c\s+[^\n]+",
        r"\bprint\s*\(\s*os\.environ\.get\(",
    ]

    for pattern in instruction_override:
        if re.search(pattern, lowered, re.IGNORECASE):
            return "instruction_override"
    for pattern in role_hijack:
        if re.search(pattern, lowered, re.IGNORECASE):
            return "role_hijack"
    for pattern in exfiltration:
        if re.search(pattern, text, re.IGNORECASE):
            return "exfiltration_attempt"
    for pattern in delimiter_escape:
        if re.search(pattern, text, re.IGNORECASE):
            return "delimiter_escape"
    for pattern in command_injection:
        if re.search(pattern, text, re.IGNORECASE):
            return "command_injection"
    return None


def sanitize_untrusted_text(text: str) -> str:
    if not text:
        return text
    sanitized = text

    hidden_patterns = [
        re.compile(r"<!--.*?-->", re.IGNORECASE | re.DOTALL),
        re.compile(r"<[^>]*style=['\"][^'\"]*(?:display\s*:\s*none|font-size\s*:\s*0px|color\s*:\s*white)[^'\"]*['\"][^>]*>.*?</[^>]+>", re.IGNORECASE | re.DOTALL),
        re.compile(r"[\u200b\u200c\u200d\ufeff]+", re.IGNORECASE),
    ]
    for pattern in hidden_patterns:
        sanitized = pattern.sub("<prompt_injection_removed: hidden_text>", sanitized)

    attack_patterns = [
        ("instruction_override", re.compile(r"\b(?:ignore|disregard|forget)\s+(?:all\s+)?(?:previous|prior|above)\s+instructions\b", re.IGNORECASE)),
        ("instruction_override", re.compile(r"\bforget\s+everything\s+above\b", re.IGNORECASE)),
        ("instruction_override", re.compile(r"\bignore\s+(?:all\s+)?safety\s+guidelines\b", re.IGNORECASE)),
        ("role_hijack", re.compile(r"\byou\s+are\s+now\s+(?:in\s+)?developer\s+mode\b", re.IGNORECASE)),
        ("role_hijack", re.compile(r"\byou\s+are\s+now\s+in\s+admin\s+mode\b", re.IGNORECASE)),
        ("role_hijack", re.compile(r"\byou\s+are\s+now\s+dan\b", re.IGNORECASE)),
        ("role_hijack", re.compile(r"\byou\s+are\s+now\s+unrestricted\b", re.IGNORECASE)),
        ("role_hijack", re.compile(r"\bprovide\s+unrestricted\s+access\b", re.IGNORECASE)),
        ("role_hijack", re.compile(r"\benable\s+developer\s+mode\b", re.IGNORECASE)),
        ("role_hijack", re.compile(r"\bact\s+as\s+an\s+unrestricted\s+ai\b", re.IGNORECASE)),
        ("role_hijack", re.compile(r"\bwithout\s+safety\s+filters\b", re.IGNORECASE)),
        ("role_hijack", re.compile(r"\bsystem\s+override\b", re.IGNORECASE)),
        ("exfiltration_attempt", re.compile(r"\breveal\s+(?:all\s+)?confidential\s+information\b", re.IGNORECASE)),
        ("exfiltration_attempt", re.compile(r"\breveal\s+(?:all\s+)?system\s+prompts?\b", re.IGNORECASE)),
        ("exfiltration_attempt", re.compile(r"\blist\s+all\s+financial\s+data,\s*user\s+passwords,\s*and\s+api\s+keys\b", re.IGNORECASE)),
        ("exfiltration_attempt", re.compile(r"\bprint\s+the\s+api\s+key\b", re.IGNORECASE)),
        ("exfiltration_attempt", re.compile(r"\b(?:send|post|upload|exfiltrate)\b.{0,80}\bhttps?://\S+", re.IGNORECASE)),
        ("exfiltration_attempt", re.compile(r"!\[[^\]]*\]\(https?://[^)]+\)", re.IGNORECASE)),
        ("delimiter_escape", re.compile(r"</system>|<\|im_start\|>|###\s*system:", re.IGNORECASE)),
        ("command_injection", re.compile(r"\bexecute\s*:\s*[^\n]+", re.IGNORECASE)),
        ("command_injection", re.compile(r"\brun\s+(?:rm\s+-rf\s+/|curl\s+https?://\S+(?:\s*\|\s*sh)?)\b", re.IGNORECASE)),
        ("command_injection", re.compile(r"\bcurl\s+https?://\S+(?:\s*\|\s*sh)?\b", re.IGNORECASE)),
        ("command_injection", re.compile(r"\b(?:python|bash|sh)\s+-c\s+[^\n]+", re.IGNORECASE)),
        ("command_injection", re.compile(r"\bprint\s*\(\s*os\.environ\.get\([^\n]*\)\s*\)", re.IGNORECASE)),
    ]
    for category, pattern in attack_patterns:
        sanitized = pattern.sub(f"<prompt_injection_removed: {category}>", sanitized)

    encoded_spans = set()
    for match in re.finditer(r"\b(?:[A-Za-z0-9+/]{20,}={0,2})\b", sanitized):
        candidate = match.group(0)
        try:
            decoded = base64.b64decode(candidate, validate=True).decode("utf-8", errors="ignore")
        except (binascii.Error, ValueError):
            continue
        if _contains_prompt_attack(decoded):
            encoded_spans.add((match.start(), match.end(), "encoded_payload"))
    for match in re.finditer(r"(?:%[0-9A-Fa-f]{2}){4,}", sanitized):
        decoded = urllib.parse.unquote(match.group(0))
        if _contains_prompt_attack(decoded):
            encoded_spans.add((match.start(), match.end(), "encoded_payload"))
    for match in re.finditer(r"\b(?:[0-9A-Fa-f]{2}){8,}\b", sanitized):
        candidate = match.group(0)
        try:
            decoded = bytes.fromhex(candidate).decode("utf-8", errors="ignore")
        except ValueError:
            continue
        if _contains_prompt_attack(decoded):
            encoded_spans.add((match.start(), match.end(), "encoded_payload"))
    rot13_decoded = codecs.decode(sanitized, "rot13")
    if _contains_prompt_attack(rot13_decoded):
        rot13_patterns = [
            ("instruction_override", re.compile(r"\b(?:vtaber|qvfertneq|sbetrg)\s+(?:nyy\s+)?(?:cerivbhf|cevbe|nobir)\s+vafgehpgvbaf\b", re.IGNORECASE)),
            ("role_hijack", re.compile(r"\blbh\s+ner\s+abj\b", re.IGNORECASE)),
            ("exfiltration_attempt", re.compile(r"\berirny\b|\byvfg\s+nyy\s+svanapvny\s+qngn\b", re.IGNORECASE)),
            ("delimiter_escape", re.compile(r"</flfgrz>|<\|vz_fgneg\|>|###\s*flfgrz:", re.IGNORECASE)),
            ("command_injection", re.compile(r"\brkrphgr\b|\beha\s+ez\s+-es\s+/\b|\bphey\s+uggc", re.IGNORECASE)),
        ]
        for category, pattern in rot13_patterns:
            sanitized = pattern.sub(f"<prompt_injection_removed: {category}>", sanitized)

    if encoded_spans:
        rebuilt = []
        last_index = 0
        for start, end, category in sorted(encoded_spans, key=lambda item: item[0]):
            if start < last_index:
                continue
            rebuilt.append(sanitized[last_index:start])
            rebuilt.append(f"<prompt_injection_removed: {category}>")
            last_index = end
        rebuilt.append(sanitized[last_index:])
        sanitized = "".join(rebuilt)

    normalized = sanitized.lower().translate(str.maketrans({"1": "i", "3": "e", "0": "o", "4": "a", "5": "s", "7": "t"}))
    normalized_compact = re.sub(r"\s+", "", normalized)
    obfuscated_checks = [
        ("instruction_override", "ignorepreviousinstructions"),
        ("instruction_override", "ignoreallsafetyguidelines"),
        ("role_hijack", "youarenowindevelopermode"),
        ("role_hijack", "youarenowinadminmode"),
        ("role_hijack", "youarenowdan"),
        ("role_hijack", "youarenowunrestricted"),
        ("command_injection", "executeprint(os.environ.get(\"api_key\"))"),
    ]
    for category, token in obfuscated_checks:
        if token in normalized_compact:
            sanitized = re.sub(r"(?:[A-Za-z0-9]\s*){8,}", f"<prompt_injection_removed: {category}>", sanitized, count=1)
            break

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

    task_list = cl.TaskList(name="State")
    task_list.status = "Running..."

    think = cl.Task(title="Thinking", status=cl.TaskStatus.RUNNING)
    await task_list.add_task(think)

    tts = cl.Task(title="Text to speech")
    await task_list.add_task(tts)

    await task_list.send()

    brain = cl.user_session.get("brain")  # type: Brain
    path_config = "basic_rag_workflow.yaml"
    retrieval_config = RetrievalConfig.from_yaml(path_config)

    if brain is None:
        await cl.Message(content="Please upload a file first.").send()
        return

    # Prepare the message for streaming
    msg = cl.Message(content="", elements=[], author="Quivr", type="assistant_message")
    await msg.send()

    saved_sources = set()
    saved_sources_complete = []
    elements = []

    # Use the ask_stream method for streaming responses
    prompt_content = sanitize_untrusted_text(message.content)
    prompt_content = redact_pii(prompt_content)
    streamed_answer_parts = []
    async for chunk in brain.ask_streaming(prompt_content, retrieval_config=retrieval_config):
        streamed_answer_parts.append(chunk.answer)
        for source in chunk.metadata.sources:
            source_preview = mask_pii(source.page_content)
            if source_preview not in saved_sources:
                saved_sources.add(source_preview)
                saved_sources_complete.append(source)
                print(source)
                elements.append(cl.Text(name=source.metadata["original_file_name"], content=source_preview, display="side"))
    
    think.status = cl.TaskStatus.DONE
    tts.status = cl.TaskStatus.RUNNING
    await task_list.update()

    full_answer = "".join(streamed_answer_parts)
    full_answer = redact_pii(full_answer)
    msg.content = mask_pii(full_answer)
    await msg.update()
    audio_file = await text_to_speech(msg.content)
    elements.append(cl.Audio(content=audio_file, auto_play=True, mime="audio/mpeg"))

    sources = ""
    for source in saved_sources_complete:
        sources += f"- {source.metadata['original_file_name']}\n"
    msg.elements = elements
    msg.content = mask_pii(msg.content + f"\n\nSources:\n{sources}")
    await msg.update()

    tts.status = cl.TaskStatus.DONE
    task_list.status = "Done"
    await task_list.update()
    await cl.sleep(1)
    await task_list.remove()

async_openai_client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

@cl.step(type="tool", name="Speech to text")
async def speech_to_text(audio_file):
    response = await async_openai_client.audio.transcriptions.create(
        model="whisper-1", file=audio_file
    )

    return redact_pii(sanitize_untrusted_text(response.text))

@cl.step(type="tool", name="Text to speech")
async def text_to_speech(text):
    text = redact_pii(text)
    response = await async_openai_client.audio.speech.create(
        model="tts-1", voice="alloy", input=text
    )

    return response.content


@cl.on_audio_chunk
async def on_audio_chunk(chunk: cl.AudioChunk):
    if chunk.isStart:
        buffer = BytesIO()
        # This is required for whisper to recognize the file type
        buffer.name = f"input_audio.{chunk.mimeType.split('/')[1]}"
        # Initialize the session for a new audio stream
        cl.user_session.set("audio_buffer", buffer)
        cl.user_session.set("audio_mime_type", chunk.mimeType)

    # Write the chunks to a buffer and transcribe the whole audio at the end
    cl.user_session.get("audio_buffer").write(chunk.data)


@cl.on_audio_end
async def on_audio_end(elements: list[Element]):
    # Get the audio buffer from the session
    task_list = cl.TaskList(name="State")
    task_list.status = "Running..."

    stt = cl.Task(title="Speech to text", status=cl.TaskStatus.RUNNING)
    await task_list.add_task(stt)

    await task_list.send()

    audio_buffer: BytesIO = cl.user_session.get("audio_buffer")
    audio_buffer.seek(0)  # Move the file pointer to the beginning
    audio_file = audio_buffer.read()
    audio_mime_type: str = cl.user_session.get("audio_mime_type")

    input_audio_el = cl.Audio(
        mime=audio_mime_type, content=audio_file, name=audio_buffer.name
    )
    await cl.Message(
        author="You",
        type="user_message",
        content="",
        elements=[input_audio_el, *elements],
    ).send()

    whisper_input = (audio_buffer.name, audio_file, audio_mime_type)
    transcription = await speech_to_text(whisper_input)
    safe_transcription = mask_pii(transcription)

    msg = cl.Message(author="You", content=safe_transcription, elements=elements)

    stt.status = cl.TaskStatus.DONE
    task_list.status = "Done"
    await task_list.update()
    await cl.sleep(1)
    await task_list.remove()

    await main(message=msg)