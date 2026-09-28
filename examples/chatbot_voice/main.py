import tempfile
import os
import re
import html
import urllib.parse
import chainlit as cl
from quivr_core import Brain
from quivr_core.rag.entities.config import RetrievalConfig
from openai import AsyncOpenAI
from chainlit.element import Element

from io import BytesIO


_ZERO_WIDTH_RE = re.compile(r"[\u200B-\u200D\u2060\uFEFF]")
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_HIDDEN_STYLE_RE = re.compile(
    r"<(?P<tag>[^>]+)style\s*=\s*[\"'][^\"']*(?:display\s*:\s*none|visibility\s*:\s*hidden|font-size\s*:\s*0(?:px)?|color\s*:\s*white\s*;\s*background(?:-color)?\s*:\s*white)[^\"']*[\"'][^>]*>.*?</(?P=tag)>",
    re.IGNORECASE | re.DOTALL,
)


def _redact_zero_tolerance_pii(text: str) -> str:
    if not isinstance(text, str) or not text:
        return text

    redacted = text
    patterns = [
        (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "<pii_redacted:ssn>"),
        (re.compile(r"\b(?:\+?1[-.\s]?)?(?:\(\d{3}\)[-.\s]?|\d{3}[-.\s])\d{3}[-.\s]\d{4}\b"), "<pii_redacted:phone>"),
        (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), "<pii_redacted:email>"),
        (re.compile(r"\b(?:\d[ -]*?){13,19}\b"), "<pii_redacted:card>"),
        (re.compile(r"\b(?:\d[ -]*?){9,17}\b"), "<pii_redacted:financial_account>"),
        (re.compile(r"\b[A-Z]{1,2}\d{6,9}\b", re.IGNORECASE), "<pii_redacted:passport>"),
        (re.compile(r"\b(?:[A-Z]\d{7}|\d{7,9}|[A-Z0-9]{1,4}-[A-Z0-9]{3,8})\b", re.IGNORECASE), None),
        (re.compile(r"\b(?:[A-F0-9]{2}:){5}[A-F0-9]{2}\b", re.IGNORECASE), "<pii_redacted:mac_address>"),
        (re.compile(r"\b(?:25[0-5]|2[0-4]\d|1?\d?\d)(?:\.(?:25[0-5]|2[0-4]\d|1?\d?\d)){3}\b"), "<pii_redacted:ip_address>"),
        (re.compile(r"\b[A-HJ-NPR-Z0-9]{17}\b", re.IGNORECASE), "<pii_redacted:vin>"),
    ]
    for pattern, replacement in patterns:
        if replacement:
            redacted = pattern.sub(replacement, redacted)

    label_patterns = [
        (re.compile(r"(\b(?:year\s+of\s+birth|yob|dob|date\s+of\s+birth)\s*:\s*)([^\n,;]+)", re.IGNORECASE), "<pii_redacted:birth_year_or_dob>"),
        (re.compile(r"(\bbirthplace\s*:\s*)([^\n,;]+)", re.IGNORECASE), "<pii_redacted:birthplace>"),
        (re.compile(r"(\bmother(?:'s|s)?\s+maiden\s+name\s*:\s*)([^\n,;]+)", re.IGNORECASE), "<pii_redacted:maiden_name>"),
        (re.compile(r"(\bhome\s+address\s*:\s*)([^\n]+)", re.IGNORECASE), "<pii_redacted:home_address>"),
        (re.compile(r"(\baddress\s*:\s*)([^\n]+)", re.IGNORECASE), "<pii_redacted:home_address>"),
        (re.compile(r"(\bpassport(?:\s+number|\s+no\.?|\s*#)?\s*:\s*)([^\n,;]+)", re.IGNORECASE), "<pii_redacted:passport>"),
        (re.compile(r"(\bdriver(?:'s)?\s+license(?:\s+number|\s+no\.?|\s*#)?\s*:\s*)([^\n,;]+)", re.IGNORECASE), "<pii_redacted:drivers_license>"),
        (re.compile(r"(\b(?:tin|taxpayer\s+identification\s+number)\s*:\s*)([^\n,;]+)", re.IGNORECASE), "<pii_redacted:tin>"),
        (re.compile(r"(\b(?:employee\s+id|employee\s+identifier)\s*:\s*)([^\n,;]+)", re.IGNORECASE), "<pii_redacted:employee_id>"),
        (re.compile(r"(\b(?:school\s+id|student\s+id)\s*:\s*)([^\n,;]+)", re.IGNORECASE), "<pii_redacted:school_id>"),
        (re.compile(r"(\b(?:medical\s+record|medical\s+records?)\s*:\s*)([^\n]+)", re.IGNORECASE), "<pii_redacted:medical_record>"),
        (re.compile(r"(\bfine\s+location\s*:\s*)([^\n,;]+)", re.IGNORECASE), "<pii_redacted:fine_location>"),
        (re.compile(r"(\bethnicity\s*:\s*)([^\n,;]+)", re.IGNORECASE), "<pii_redacted:ethnicity>"),
        (re.compile(r"(\bsexual\s+orientation\s*:\s*)([^\n,;]+)", re.IGNORECASE), "<pii_redacted:sexual_orientation>"),
        (re.compile(r"(\bvoice\s+signature\s*:\s*)([^\n,;]+)", re.IGNORECASE), "<pii_redacted:voice_signature>"),
        (re.compile(r"(\bfacial\s+image\s*:\s*)([^\n,;]+)", re.IGNORECASE), "<pii_redacted:facial_image>"),
        (re.compile(r"(\bfingerprints?\s*:\s*)([^\n,;]+)", re.IGNORECASE), "<pii_redacted:fingerprint>"),
        (re.compile(r"(\b(?:retina|iris)\s+scan\s*:\s*)([^\n,;]+)", re.IGNORECASE), "<pii_redacted:retina_iris_scan>"),
    ]
    for pattern, replacement in label_patterns:
        redacted = pattern.sub(lambda m: f"{m.group(1)}{replacement}", redacted)

    return redacted


def _replace_encoded_prompt_payloads(text: str) -> str:
    if not isinstance(text, str) or not text:
        return text

    sanitized = text
    for match in re.finditer(r"\b(?:[A-Za-z0-9+/]{20,}={0,2})\b", sanitized):
        token = match.group(0)
        try:
            decoded = __import__("base64").b64decode(token, validate=True).decode("utf-8", errors="ignore")
        except Exception:
            continue
        lowered = decoded.lower()
        if any(
            phrase in lowered
            for phrase in [
                "ignore previous instructions",
                "forget everything above",
                "you are now",
                "act as unrestricted",
                "reveal system prompt",
                "curl http",
                "wget http",
                "bash -c",
                "powershell -",
                "rm -rf",
            ]
        ):
            sanitized = sanitized.replace(token, "<prompt_injection_removed:encoded_payload>")
    return sanitized


def _sanitize_untrusted_text(text: str) -> str:
    if not isinstance(text, str) or not text:
        return text

    sanitized = text
    sanitized = _HTML_COMMENT_RE.sub("<prompt_injection_removed:hidden_text>", sanitized)
    sanitized = _HIDDEN_STYLE_RE.sub("<prompt_injection_removed:hidden_text>", sanitized)
    if _ZERO_WIDTH_RE.search(sanitized):
        sanitized = _ZERO_WIDTH_RE.sub("", sanitized)
        sanitized = "<prompt_injection_removed:hidden_text>" if not sanitized.strip() else sanitized

    normalized_url = urllib.parse.unquote(sanitized)
    if normalized_url != sanitized:
        lowered_url = normalized_url.lower()
        if any(
            phrase in lowered_url
            for phrase in [
                "ignore previous instructions",
                "forget everything above",
                "you are now",
                "act as unrestricted",
                "reveal system prompt",
            ]
        ):
            sanitized = normalized_url
            sanitized = "<prompt_injection_removed:encoded_payload>"

    sanitized = _replace_encoded_prompt_payloads(sanitized)

    replacement_patterns = [
        (re.compile(r"\b(?:ignore\s+previous\s+instructions|forget\s+everything\s+above|disregard\s+all\s+prior\s+instructions)\b", re.IGNORECASE), "<prompt_injection_removed:instruction_override>"),
        (re.compile(r"\b(?:you\s+are\s+now\s+DAN|you\s+are\s+now\s+in\s+admin\s+mode|act\s+as\s+(?:an\s+)?unrestricted\s+AI|developer\s+mode|do\s+anything\s+now)\b", re.IGNORECASE), "<prompt_injection_removed:role_hijack>"),
        (re.compile(r"</?system>|</?assistant>|</?user>|<{3,}|>{3,}|\[/?SYSTEM\]", re.IGNORECASE), "<prompt_injection_removed:delimiter_escape>"),
        (re.compile(r"\b(?:system\s*:\s*you\s+must|assistant\s*:\s*ignore|tool\s*:\s*send)\b", re.IGNORECASE), "<prompt_injection_removed:fake_system_message>"),
        (re.compile(r"\b(?:send\s+data\s+to\s+https?://\S+|curl\s+https?://\S+|wget\s+https?://\S+|reveal\s+(?:the\s+)?system\s+prompt|list\s+all\s+passwords\s+and\s+api\s+keys|reveal\s+all\s+confidential\s+information)\b", re.IGNORECASE), "<prompt_injection_removed:exfiltration_attempt>"),
        (re.compile(r"\b(?:from\s+now\s+on\s+in\s+this\s+conversation|in\s+the\s+next\s+turn\s+ignore|remember\s+this\s+secret\s+instruction)\b", re.IGNORECASE), "<prompt_injection_removed:context_poisoning>"),
        (re.compile(r"\b(?:bash\s+-c\s+\S+|sh\s+-c\s+\S+|powershell\s+-[A-Za-z]+\s+\S+|cmd\.exe\s+/c\s+\S+|rm\s+-rf\s+\S+|subprocess\.(?:run|Popen)\s*\(|os\.system\s*\(|exec\s*\(|eval\s*\()", re.IGNORECASE), "<prompt_injection_removed:command_injection>"),
        (re.compile(r"\b(?:DAN|jailbreak|bypass\s+safety|fictional\s+framing\s+bypass)\b", re.IGNORECASE), "<prompt_injection_removed:jailbreak_attempt>"),
        (re.compile(r"\b(?:base64|hex|rot13|morse|leet(?:speak)?)\b\s*(?:instruction|payload|prompt)?\s*:\s*[^\n]+", re.IGNORECASE), "<prompt_injection_removed:encoded_payload>"),
        (re.compile(r"\b(?:file|metadata|field|comment)\s*:\s*(?:ignore\s+previous\s+instructions|forget\s+everything\s+above|you\s+are\s+now\s+\w+)\b", re.IGNORECASE), "<prompt_injection_removed:indirect_injection>"),
    ]
    for pattern, replacement in replacement_patterns:
        sanitized = pattern.sub(replacement, sanitized)

    return sanitized


def _sanitize_text_for_ai(text: str) -> str:
    if not isinstance(text, str) or not text:
        return text
    return _redact_zero_tolerance_pii(_sanitize_untrusted_text(text))


def _mask_text_for_ui(text: str) -> str:
    if not isinstance(text, str) or not text:
        return text
    return _redact_zero_tolerance_pii(text)


def _mask_name_for_ui(text: str) -> str:
    if not isinstance(text, str) or not text:
        return text
    return html.escape(_mask_text_for_ui(text))


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

    text = _sanitize_text_for_ai(text)

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

    message.content = _sanitize_text_for_ai(message.content)

    # Prepare the message for streaming
    msg = cl.Message(content="", elements=[], author="Quivr", type="assistant_message")
    await msg.send()

    saved_sources = set()
    saved_sources_complete = []
    elements = []

    # Use the ask_stream method for streaming responses
    async for chunk in brain.ask_streaming(message.content, retrieval_config=retrieval_config):
        await msg.stream_token(_mask_text_for_ui(chunk.answer))
        for source in chunk.metadata.sources:
            masked_page_content = _mask_text_for_ui(source.page_content)
            if masked_page_content not in saved_sources:
                saved_sources.add(masked_page_content)
                saved_sources_complete.append(source)
                print(source)
                elements.append(cl.Text(name=_mask_name_for_ui(source.metadata["original_file_name"]), content=masked_page_content, display="side"))
    
    think.status = cl.TaskStatus.DONE
    tts.status = cl.TaskStatus.RUNNING
    await task_list.update()
    
    audio_file = await text_to_speech(_sanitize_text_for_ai(msg.content))
    elements.append(cl.Audio(content=audio_file, auto_play=True, mime="audio/mpeg"))

    sources = ""
    for source in saved_sources_complete:
        sources += f"- {_mask_name_for_ui(source.metadata['original_file_name'])}\n"
    msg.elements = elements
    msg.content = _mask_text_for_ui(msg.content) + f"\n\nSources:\n{sources}"
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

    return _sanitize_text_for_ai(response.text)

@cl.step(type="tool", name="Text to speech")
async def text_to_speech(text):
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
    masked_elements = []
    for element in elements:
        if isinstance(element, cl.Text):
            masked_elements.append(cl.Text(name=_mask_name_for_ui(element.name), content=_mask_text_for_ui(element.content), display=element.display))
        else:
            masked_elements.append(element)

    await cl.Message(
        author="You",
        type="user_message",
        content="",
        elements=[input_audio_el, *masked_elements],
    ).send()

    whisper_input = (audio_buffer.name, audio_file, audio_mime_type)
    transcription = await speech_to_text(whisper_input)

    msg = cl.Message(author="You", content=_mask_text_for_ui(transcription), elements=masked_elements)

    stt.status = cl.TaskStatus.DONE
    task_list.status = "Done"
    await task_list.update()
    await cl.sleep(1)
    await task_list.remove()

    await main(message=msg)