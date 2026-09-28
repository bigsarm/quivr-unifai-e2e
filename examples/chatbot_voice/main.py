import tempfile
import os
import chainlit as cl
from quivr_core import Brain
from quivr_core.rag.entities.config import RetrievalConfig
from openai import AsyncOpenAI
from chainlit.element import Element

from io import BytesIO
import base64
import re
import urllib.parse


ZERO_WIDTH_RE = re.compile(r"[\u200b\u200c\u200d\ufeff]")
PII_PATTERNS = [
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "<redacted_ssn>"),
    (re.compile(r"\b(?:19|20)\d{2}\b"), "<redacted_year_of_birth>"),
    (re.compile(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[A-Za-z]{2,}\b"), "<redacted_email>"),
    (re.compile(r"\b(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})\b"), "<redacted_phone>"),
    (re.compile(r"\b\d{13,19}\b"), "<redacted_financial_number>"),
    (re.compile(r"\b(?:\d[ -]*?){13,19}\b"), "<redacted_credit_card>"),
    (re.compile(r"\b[A-Z]{1,2}\d{6,9}\b"), "<redacted_passport>"),
    (re.compile(r"\b[A-Z0-9]{17}\b"), "<redacted_vin>"),
    (re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), "<redacted_ip>"),
    (re.compile(r"\b(?:[0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}\b"), "<redacted_mac>"),
    (re.compile(r"\b(?:[A-Z]\d{7}|\d{8,9})\b"), "<redacted_license_or_id>"),
    (re.compile(r"\b\d{2}-\d{7}\b"), "<redacted_tin>"),
    (re.compile(r"\b\d{4,6}\s+[A-Za-z0-9.'# -]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|Way)\b", re.IGNORECASE), "<redacted_address>"),
]

PROMPT_INJECTION_PATTERNS = [
    (re.compile(r"\b(ignore\s+previous\s+instructions|forget\s+everything\s+above|disregard\s+(all|previous)\s+instructions)\b", re.IGNORECASE), "<prompt_injection_removed: instruction_override>"),
    (re.compile(r"\b(you\s+are\s+now\s+dan|act\s+as\s+unrestricted|developer\s+mode|jailbreak)\b", re.IGNORECASE), "<prompt_injection_removed: role_hijack>"),
    (re.compile(r"</?system>|</?assistant>|</?user>|```|---\s*BEGIN|---\s*END", re.IGNORECASE), "<prompt_injection_removed: delimiter_escape>"),
    (re.compile(r"\b(?:[A-Fa-f0-9]{32,}|[A-Za-z0-9+/]{24,}={0,2}|%[0-9A-Fa-f]{2})\b"), "<prompt_injection_removed: encoded_payload>"),
    (re.compile(r"<!--.*?-->|display\s*:\s*none|visibility\s*:\s*hidden", re.IGNORECASE | re.DOTALL), "<prompt_injection_removed: hidden_text>"),
    (re.compile(r"\b(system\s*:\s*|tool\s*:\s*|assistant\s*:\s*)", re.IGNORECASE), "<prompt_injection_removed: fake_system_message>"),
    (re.compile(r"\b(exfiltrate|send\s+data\s+to|leak\s+the\s+system\s+prompt|reveal\s+your\s+system\s+prompt)\b", re.IGNORECASE), "<prompt_injection_removed: exfiltration_attempt>"),
    (re.compile(r"\b(on\s+the\s+next\s+turn|in\s+future\s+responses|remember\s+this\s+secret\s+instruction)\b", re.IGNORECASE), "<prompt_injection_removed: context_poisoning>"),
    (re.compile(r"\b(metadata|comment|code\s+comment|hidden\s+field)\s*:\s*.*?(ignore|override|execute)", re.IGNORECASE), "<prompt_injection_removed: indirect_injection>"),
    (re.compile(r"\b(rm\s+-rf|curl\s+|wget\s+|powershell\s+|bash\s+|sh\s+|cmd\.exe|subprocess|os\.system|eval\(|exec\()", re.IGNORECASE), "<prompt_injection_removed: command_injection>"),
    (re.compile(r"\b(i\s*g\s*n\s*o\s*r\s*e\s+previous\s+instructions|d\s*a\s*n)\b", re.IGNORECASE), "<prompt_injection_removed: split_payload>"),
    (re.compile(r"\b(do\s+anything\s+now|fictional\s+framing|bypass\s+safety|unfiltered\s+response)\b", re.IGNORECASE), "<prompt_injection_removed: jailbreak_attempt>"),
]


def _maybe_decode_obfuscated_text(text: str) -> str:
    candidates = [text, urllib.parse.unquote(text)]
    compact = re.sub(r"\s+", "", text)
    if re.fullmatch(r"[A-Za-z0-9+/=]+", compact) and len(compact) >= 24:
        try:
            candidates.append(base64.b64decode(compact, validate=True).decode("utf-8", errors="ignore"))
        except Exception:
            pass
    return "\n".join(candidate for candidate in candidates if candidate)


def _sanitize_ai_input(text: str) -> str:
    if not text:
        return text
    sanitized = ZERO_WIDTH_RE.sub("<prompt_injection_removed: hidden_text>", text)
    decoded_view = _maybe_decode_obfuscated_text(sanitized)
    for pattern, replacement in PROMPT_INJECTION_PATTERNS:
        if pattern.search(decoded_view) or pattern.search(sanitized):
            sanitized = pattern.sub(replacement, sanitized)
    return sanitized


def _redact_pii(text: str) -> str:
    if not text:
        return text
    redacted = text
    for pattern, replacement in PII_PATTERNS:
        redacted = pattern.sub(replacement, redacted)
    return redacted


def _prepare_ai_text(text: str) -> str:
    return _redact_pii(_sanitize_ai_input(text))


def _mask_ui_text(text: str) -> str:
    return _redact_pii(_sanitize_ai_input(text))


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

    await cl.Message(content="Notice: the configured audio models are not in the organization registry. Please replace them with an approved model before production use.").send()

    with open(file.path, "r", encoding="utf-8") as f:
        text = f.read()

    text = _prepare_ai_text(text)

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
    user_message = _prepare_ai_text(message.content)
    async for chunk in brain.ask_streaming(user_message, retrieval_config=retrieval_config):
        await msg.stream_token(_mask_ui_text(chunk.answer))
        for source in chunk.metadata.sources:
            sanitized_source_content = _mask_ui_text(source.page_content)
            if sanitized_source_content not in saved_sources:
                saved_sources.add(sanitized_source_content)
                saved_sources_complete.append(source)
                print(source)
                elements.append(cl.Text(name=source.metadata["original_file_name"], content=sanitized_source_content, display="side"))
    
    think.status = cl.TaskStatus.DONE
    tts.status = cl.TaskStatus.RUNNING
    await task_list.update()
    
    msg.content = _mask_ui_text(msg.content)
    audio_file = await text_to_speech(msg.content)
    elements.append(cl.Audio(content=audio_file, auto_play=True, mime="audio/mpeg"))

    sources = ""
    for source in saved_sources_complete:
        sources += f"- {source.metadata['original_file_name']}\n"
    msg.elements = elements
    msg.content = msg.content + f"\n\nSources:\n{sources}"
    await msg.update()

    tts.status = cl.TaskStatus.DONE
    task_list.status = "Done"
    await task_list.update()
    await cl.sleep(1)
    await task_list.remove()

async_openai_client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

@cl.step(type="tool", name="Speech to text")
async def speech_to_text(audio_file):
    await cl.Message(content="Notice: speech-to-text is configured with an unapproved model. Please replace it with an approved model.").send()
    response = await async_openai_client.audio.transcriptions.create(
        model="whisper-1", file=audio_file
    )

    return response.text

@cl.step(type="tool", name="Text to speech")
async def text_to_speech(text):
    await cl.Message(content="Notice: text-to-speech is configured with an unapproved model. Please replace it with an approved model.").send()
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
        elements=elements,
    ).send()

    whisper_input = (audio_buffer.name, audio_file, audio_mime_type)
    transcription = await speech_to_text(whisper_input)
    transcription = _prepare_ai_text(transcription)

    msg = cl.Message(author="You", content=_mask_ui_text(transcription), elements=elements)

    stt.status = cl.TaskStatus.DONE
    task_list.status = "Done"
    await task_list.update()
    await cl.sleep(1)
    await task_list.remove()

    await main(message=msg)