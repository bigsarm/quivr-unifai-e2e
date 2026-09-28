from flask import Flask, render_template, request, jsonify, session
import openai
import base64
import os
import requests
from dotenv import load_dotenv
from quivr_core import Brain
from quivr_core.rag.entities.config import RetrievalConfig
from tempfile import NamedTemporaryFile
from werkzeug.utils import secure_filename
from asyncio import to_thread
import asyncio
import re
import binascii
import codecs
import html
import urllib.parse


UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"txt"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.secret_key = "secret"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["CACHE_TYPE"] = "SimpleCache"  # In-memory cache for development
app.config["CACHE_DEFAULT_TIMEOUT"] = 60 * 60  # 1 hour cache timeout
load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

brains = {}


@app.route("/")
def index():
    return render_template("index.html")


def run_in_event_loop(func, *args, **kwargs):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    if asyncio.iscoroutinefunction(func):
        result = loop.run_until_complete(func(*args, **kwargs))
    else:
        result = func(*args, **kwargs)
    loop.close()
    return result


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _replace_matches(text, patterns, label):
    for pattern in patterns:
        text = re.sub(pattern, label, text, flags=re.IGNORECASE)
    return text


def _normalize_for_obfuscation(text):
    translation = str.maketrans({"1": "i", "3": "e", "0": "o", "4": "a", "5": "s", "7": "t"})
    normalized_chars = []
    positions = []
    for index, char in enumerate(text):
        if char.isalnum():
            normalized_chars.append(char.lower().translate(translation))
            positions.append(index)
    return "".join(normalized_chars), positions


def _replace_obfuscated_attack(text, compact_pattern, label):
    normalized, positions = _normalize_for_obfuscation(text)
    updated_text = text
    offset = 0
    for match in re.finditer(compact_pattern, normalized, flags=re.IGNORECASE):
        start = positions[match.start()]
        end = positions[match.end() - 1] + 1
        updated_text = updated_text[: start + offset] + label + updated_text[end + offset :]
        offset += len(label) - (end - start)
    return updated_text


def _decoded_text_is_attack(decoded_text):
    attack_patterns = [
        r"(?i)\b(?:ignore|disregard|forget)\s+(?:all\s+)?(?:previous|prior|above)\s+instructions\b",
        r"(?i)\bforget\s+everything\s+above\b",
        r"(?i)\bignore\s+(?:all\s+)?safety\s+guidelines\b",
        r"(?i)\byou\s+are\s+now\s+(?:in\s+)?(?:developer|admin)\s+mode\b",
        r"(?i)\byou\s+are\s+now\s+(?:in\s+)?DAN\b",
        r"(?i)\byou\s+are\s+now\s+unrestricted\b",
        r"(?i)\bprovide\s+unrestricted\s+access\b",
        r"(?i)\benable\s+developer\s+mode\b",
        r"(?i)\bact\s+as\s+an\s+unrestricted\s+ai\b",
        r"(?i)\bwithout\s+safety\s+filters\b",
        r"(?i)\bSYSTEM\s+OVERRIDE\b",
        r"(?i)</system>|<\|im_start\|>|###\s*system:",
        r"(?i)\breveal\s+(?:all\s+)?confidential\s+information\b",
        r"(?i)\breveal\s+(?:all\s+)?system\s+prompts?\b",
        r"(?i)\blist\s+all\s+financial\s+data,\s*user\s+passwords,\s*and\s*api\s+keys\b",
        r"(?i)\bprint\s+the\s+api\s+key\b",
        r"(?i)\b(?:send|post|upload|exfiltrate)\b.{0,80}\bhttps?://\S+",
        r"(?i)\b(?:execute|run)\s+(?:rm\s+-rf\s+/|curl\s+https?://\S+\s*\|\s*sh|print\s*\(\s*os\.environ\.get\()",
    ]
    return any(re.search(pattern, decoded_text) for pattern in attack_patterns)


def sanitize_untrusted_text(text):
    if not text:
        return text

    text = re.sub(r"<!--.*?-->", "<prompt_injection_removed: hidden_text>", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(
        r"<[^>]+style\s*=\s*[\"'][^\"']*(?:display\s*:\s*none|visibility\s*:\s*hidden|font-size\s*:\s*0(?:px)?|color\s*:\s*#?fff(?:fff)?)[^\"']*[\"'][^>]*>.*?</[^>]+>",
        "<prompt_injection_removed: hidden_text>",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    text = re.sub(r"[\u200B-\u200D\uFEFF]+", "<prompt_injection_removed: hidden_text>", text)

    text = re.sub(
        r"(?i)\b(?:ignore|disregard|forget)\s+(?:all\s+)?(?:previous|prior|above)\s+instructions\b",
        "<prompt_injection_removed: instruction_override>",
        text,
    )
    text = re.sub(
        r"(?i)\bforget\s+everything\s+above\b",
        "<prompt_injection_removed: instruction_override>",
        text,
    )
    text = re.sub(
        r"(?i)\bignore\s+(?:all\s+)?safety\s+guidelines\b",
        "<prompt_injection_removed: instruction_override>",
        text,
    )

    text = re.sub(
        r"(?i)\byou\s+are\s+now\s+(?:in\s+)?(?:developer|admin)\s+mode\b",
        "<prompt_injection_removed: role_hijack>",
        text,
    )
    text = re.sub(
        r"(?i)\byou\s+are\s+now\s+(?:in\s+)?DAN\b",
        "<prompt_injection_removed: role_hijack>",
        text,
    )
    text = re.sub(
        r"(?i)\byou\s+are\s+now\s+unrestricted\b|\bprovide\s+unrestricted\s+access\b|\benable\s+developer\s+mode\b|\bact\s+as\s+an\s+unrestricted\s+ai\b|\bwithout\s+safety\s+filters\b|\bSYSTEM\s+OVERRIDE\b",
        "<prompt_injection_removed: role_hijack>",
        text,
    )

    text = re.sub(
        r"(?i)\breveal\s+(?:all\s+)?confidential\s+information\b|\breveal\s+(?:all\s+)?system\s+prompts?\b|\blist\s+all\s+financial\s+data,\s*user\s+passwords,\s*and\s*api\s+keys\b|\bprint\s+the\s+api\s+key\b|\b(?:send|post|upload|exfiltrate)\b.{0,80}\bhttps?://\S+|!\[[^\]]*\]\([^)]*data:[^)]*\)",
        "<prompt_injection_removed: exfiltration_attempt>",
        text,
    )

    text = re.sub(
        r"(?i)</system>|<\|im_start\|>|###\s*system:",
        "<prompt_injection_removed: delimiter_escape>",
        text,
    )

    text = re.sub(
        r"(?i)\b(?:execute|run)\s*:\s*(?:print\s*\(\s*os\.environ\.get\([^\n]*\)|rm\s+-rf\s+/|curl\s+https?://\S+\s*\|\s*sh)\b|\b(?:execute|run)\s+(?:print\s*\(\s*os\.environ\.get\([^\n]*\)|rm\s+-rf\s+/|curl\s+https?://\S+\s*\|\s*sh)\b",
        "<prompt_injection_removed: command_injection>",
        text,
    )

    encoded_candidates = set(re.findall(r"\b(?:[A-Za-z0-9+/]{16,}={0,2}|%[0-9A-Fa-f]{2}(?:%[0-9A-Fa-f]{2}){5,}|[0-9A-Fa-f]{20,})\b", text))
    for candidate in encoded_candidates:
        decoded_values = []
        try:
            decoded_values.append(base64.b64decode(candidate, validate=True).decode("utf-8", errors="ignore"))
        except (binascii.Error, ValueError):
            pass
        try:
            decoded_values.append(urllib.parse.unquote(candidate))
        except Exception:
            pass
        if len(candidate) % 2 == 0:
            try:
                decoded_values.append(bytes.fromhex(candidate).decode("utf-8", errors="ignore"))
            except ValueError:
                pass
        try:
            decoded_values.append(codecs.decode(candidate, "rot13"))
        except Exception:
            pass
        if any(decoded and _decoded_text_is_attack(decoded) for decoded in decoded_values):
            text = text.replace(candidate, "<prompt_injection_removed: encoded_payload>")

    text = _replace_obfuscated_attack(
        text,
        r"ignore(?:all)?(?:previous|prior|above)instructions|forgeteverythingabove|ignore(?:all)?safetyguidelines",
        "<prompt_injection_removed: instruction_override>",
    )
    text = _replace_obfuscated_attack(
        text,
        r"youarenow(?:in)?developermode|youarenow(?:in)?adminmode|youarenow(?:in)?dan|youarenowunrestricted|provideunrestrictedaccess|enabledevelopermode|actasanunrestrictedai|withoutsafetyfilters|systemoverride",
        "<prompt_injection_removed: role_hijack>",
    )

    text = re.sub(r"\b\d{3}[- ]\d{2}[- ]\d{4}\b", "<redacted:ssn>", text)
    text = re.sub(r"(?:\+1[ .-]?)?(?:\(\d{3}\)|\b\d{3})[ .-]?\d{3}[ .-]?\d{4}\b", "<redacted:phone>", text)
    text = re.sub(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b", "<redacted:email>", text)
    text = re.sub(r"\b\d{1,5}\s+(?:[A-Z][a-z]+\s){1,3}(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|Way)\b\.?(?:,\s*[A-Z][a-z]+(?:\s[A-Z][a-z]+)*)?(?:,\s*[A-Z]{2}\b(?:\s+\d{5}(?:-\d{4})?)?)?(?:,\s*(?:USA|United States)\b)?", "<redacted:address>", text)
    text = re.sub(r"\b(?:\d[ -]*?){13,19}\b", "<redacted:credit_card>", text)
    text = re.sub(r"(?i)\b(?:DOB|date of birth|born(?: on| in)?)\s*:?\s*(?:\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4}|(?:19|20)\d{2})\b", lambda m: re.sub(r"(:?\s*)(?:\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4}|(?:19|20)\d{2})\b", r"\1<redacted:dob>", m.group(0)), text)
    text = re.sub(r"(?i)\bPassport(?:\s*(?:No\.?|number|#))?\s*:?\s*(?=[A-Z0-9]*\d)[A-Z0-9]{6,9}\b", lambda m: re.sub(r"(?i)(Passport(?:\s*(?:No\.?|number|#))?\s*:?\s*)(?=[A-Z0-9]*\d)[A-Z0-9]{6,9}\b", r"\1<redacted:passport>", m.group(0)), text)
    text = re.sub(r"(?i)\b(?:Driver'?s License|Drivers License|Driver License)(?:\s*(?:No\.?|number|#))?\s*:?\s*[A-Z0-9-]{4,20}\b", lambda m: re.sub(r"(?i)((?:Driver'?s License|Drivers License|Driver License)(?:\s*(?:No\.?|number|#))?\s*:?\s*)[A-Z0-9-]{4,20}\b", r"\1<redacted:drivers_license>", m.group(0)), text)
    text = re.sub(r"(?i)\b(?:Taxpayer Identification Number|TIN|Tax ID)(?:\s*(?:No\.?|number|#))?\s*:?\s*[A-Z0-9-]{6,20}\b", lambda m: re.sub(r"(?i)((?:Taxpayer Identification Number|TIN|Tax ID)(?:\s*(?:No\.?|number|#))?\s*:?\s*)[A-Z0-9-]{6,20}\b", r"\1<redacted:tax_id>", m.group(0)), text)
    text = re.sub(r"(?i)\b(?:Financial Account Number|Account Number|Acct(?:ount)?(?: No\.?| Number)?)(?:\s*(?:No\.?|number|#))?\s*:?\s*[A-Z0-9-]{6,20}\b", lambda m: re.sub(r"(?i)((?:Financial Account Number|Account Number|Acct(?:ount)?(?: No\.?| Number)?)(?:\s*(?:No\.?|number|#))?\s*:?\s*)[A-Z0-9-]{6,20}\b", r"\1<redacted:account_number>", m.group(0)), text)
    text = re.sub(r"(?i)\bEmployee ID\s*:?\s*[A-Z0-9-]{2,20}\b", lambda m: re.sub(r"(?i)(Employee ID\s*:?\s*)[A-Z0-9-]{2,20}\b", r"\1<redacted:employee_id>", m.group(0)), text)
    text = re.sub(r"(?i)\bSchool ID\s*:?\s*[A-Z0-9-]{2,20}\b", lambda m: re.sub(r"(?i)(School ID\s*:?\s*)[A-Z0-9-]{2,20}\b", r"\1<redacted:school_id>", m.group(0)), text)
    text = re.sub(r"(?i)\bVIN\s*:?\s*[A-HJ-NPR-Z0-9]{17}\b", lambda m: re.sub(r"(?i)(VIN\s*:?\s*)[A-HJ-NPR-Z0-9]{17}\b", r"\1<redacted:vin>", m.group(0)), text)
    text = re.sub(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", "<redacted:ip_address>", text)
    text = re.sub(r"\b(?:[0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}\b", "<redacted:mac_address>", text)
    text = re.sub(r"(?i)\bBirthplace\s*:?\s*[^,;\n]+", lambda m: re.sub(r"(?i)(Birthplace\s*:?\s*)[^,;\n]+", r"\1<redacted:birthplace>", m.group(0)), text)
    text = re.sub(r"(?i)\bMother'?s Maiden Name\s*:?\s*[^,;\n]+", lambda m: re.sub(r"(?i)(Mother'?s Maiden Name\s*:?\s*)[^,;\n]+", r"\1<redacted:maiden_name>", m.group(0)), text)
    text = re.sub(r"(?i)\bMedical Records?\s*:?\s*[^\n]+", lambda m: re.sub(r"(?i)(Medical Records?\s*:?\s*)[^\n]+", r"\1<redacted:medical>", m.group(0)), text)
    text = re.sub(r"(?i)\b(?:Fine Location|Location)\s*:?\s*[^\n]+", lambda m: re.sub(r"(?i)(?:Fine Location|Location)(\s*:?\s*)[^\n]+", r"\1<redacted:location>", m.group(0)), text)
    text = re.sub(r"(?i)\bEthnicity\s*:?\s*[^,;\n]+", lambda m: re.sub(r"(?i)(Ethnicity\s*:?\s*)[^,;\n]+", r"\1<redacted:ethnicity>", m.group(0)), text)
    text = re.sub(r"(?i)\bSexual Orientation\s*:?\s*[^,;\n]+", lambda m: re.sub(r"(?i)(Sexual Orientation\s*:?\s*)[^,;\n]+", r"\1<redacted:sexual_orientation>", m.group(0)), text)

    return text


def sanitize_uploaded_file(filepath):
    with open(filepath, "r", encoding="utf-8", errors="ignore") as uploaded_file:
        content = uploaded_file.read()
    sanitized_content = sanitize_untrusted_text(content)
    with open(filepath, "w", encoding="utf-8") as uploaded_file:
        uploaded_file.write(sanitized_content)


@app.route("/upload", methods=["POST"])
async def upload_file():
    if "file" not in request.files:
        return "No file part", 400

    file = request.files["file"]

    if file.filename == "":
        return "No selected file", 400
    if not (file and file.filename and allowed_file(file.filename)):
        return "Invalid file type", 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)
    sanitize_uploaded_file(filepath)

    print(f"File uploaded and saved at: {filepath}")

    print("Creating brain instance...")

    brain: Brain = await to_thread(
        run_in_event_loop, Brain.from_files, name="user_brain", file_paths=[filepath]
    )

    # Store brain instance in cache
    session_id = session.sid if hasattr(session, "sid") else os.urandom(16).hex()
    session["session_id"] = session_id
    # cache.set(session_id, brain)  # Store the brain instance in the cache
    brains[session_id] = brain
    print(f"Brain instance created and stored in cache for session ID: {session_id}")

    return jsonify({"message": "Brain created successfully"})


@app.route("/ask", methods=["POST"])
async def ask():
    if "audio_data" not in request.files:
        return "Missing audio data", 400

    # Retrieve the brain instance from the cache using the session ID
    session_id = session.get("session_id")
    if not session_id:
        return "Session ID not found. Upload a file first.", 400

    brain = brains.get(session_id)
    if not brain:
        return "Brain instance not found in dict. Upload a file first.", 400

    print("Brain instance loaded from cache.")

    print("Speech to text...")
    audio_file = request.files["audio_data"]
    transcript = transcribe_audio_file(audio_file)
    transcript = sanitize_untrusted_text(transcript)
    print("Transcript result: ", transcript)

    print("Getting response...")
    quivr_response = await to_thread(run_in_event_loop, brain.ask, transcript)
    quivr_response.answer = sanitize_untrusted_text(quivr_response.answer)

    print("Text to speech...")
    audio_base64 = synthesize_speech(quivr_response.answer)

    print("Done")
    return jsonify({"audio_base64": audio_base64})


def transcribe_audio_file(audio_file):
    with NamedTemporaryFile(suffix=".webm", delete=False) as temp_audio_file:
        audio_file.save(temp_audio_file)
        temp_audio_file_path = temp_audio_file.name

    try:
        with open(temp_audio_file_path, "rb") as f:
            transcript_response = openai.audio.transcriptions.create(
                model="whisper-1", file=f
            )
        transcript = transcript_response.text
        transcript = sanitize_untrusted_text(transcript)
    finally:
        os.unlink(temp_audio_file_path)

    return transcript


def synthesize_speech(text):
    text = sanitize_untrusted_text(text)
    speech_response = openai.audio.speech.create(
        model="tts-1", voice="nova", input=text
    )
    audio_content = speech_response.content
    audio_base64 = base64.b64encode(audio_content).decode("utf-8")
    return audio_base64


if __name__ == "__main__":
    app.run(debug=True)
