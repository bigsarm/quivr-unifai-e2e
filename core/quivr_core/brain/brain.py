import asyncio
import codecs
import logging
import os
import re
import urllib.parse
from pathlib import Path
from pprint import PrettyPrinter
from typing import Any, AsyncGenerator, Callable, Dict, Self, Type, Union
from uuid import UUID, uuid4

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.vectorstores import VectorStore
from langchain_openai import OpenAIEmbeddings
from rich.console import Console
from rich.panel import Panel

from quivr_core.brain.info import BrainInfo, ChatHistoryInfo
from quivr_core.brain.serialization import (
    BrainSerialized,
    EmbedderConfig,
    FAISSConfig,
    LocalStorageConfig,
    TransparentStorageConfig,
)
from quivr_core.files.file import load_qfile
from quivr_core.llm import LLMEndpoint
from quivr_core.processor.registry import get_processor_class
from quivr_core.rag.entities.chat import ChatHistory
from quivr_core.rag.entities.config import RetrievalConfig
from quivr_core.rag.entities.models import (
    LangchainMetadata,
    ParsedRAGChunkResponse,
    ParsedRAGResponse,
    QuivrKnowledge,
    SearchResult,
)
from quivr_core.rag.quivr_rag import QuivrQARAG
from quivr_core.rag.quivr_rag_langgraph import QuivrQARAGLangGraph
from quivr_core.storage.local_storage import LocalStorage, TransparentStorage
from quivr_core.storage.storage_base import StorageBase

from .brain_defaults import build_default_vectordb, default_embedder, default_llm

logger = logging.getLogger("quivr_core")


_HIDDEN_TEXT_PATTERNS = [
    re.compile(r"<!--.*?-->", re.IGNORECASE | re.DOTALL),
    re.compile(
        r"<(?P<tag>[a-zA-Z][^>\s/]*)\b[^>]*style\s*=\s*['\"][^'\"]*(?:display\s*:\s*none|visibility\s*:\s*hidden|font-size\s*:\s*0(?:px|em|rem|%)?|color\s*:\s*#(?:fff|ffffff)|color\s*:\s*white)[^'\"]*['\"][^>]*>.*?</(?P=tag)>",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r"<(?P<tag>[a-zA-Z][^>\s/]*)\b[^>]*(?:hidden|aria-hidden\s*=\s*['\"]?true['\"]?)[^>]*>.*?</(?P=tag)>",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(r"[\u200B-\u200D\uFEFF]+"),
]

_PROMPT_INJECTION_PATTERNS = [
    (
        "instruction_override",
        re.compile(
            r"(?i)\b(?:ignore|disregard|forget)\s+(?:all\s+)?(?:previous|prior|above)\s+instructions\b"
        ),
    ),
    (
        "instruction_override",
        re.compile(r"(?i)\bforget\s+everything\s+above\b"),
    ),
    (
        "instruction_override",
        re.compile(r"(?i)\bignore\s+(?:all\s+)?safety\s+guidelines\b"),
    ),
    (
        "role_hijack",
        re.compile(
            r"(?i)\byou\s+are\s+now\s+(?:in\s+)?(?:developer\s+mode|admin\s+mode|DAN|unrestricted)\b"
        ),
    ),
    (
        "role_hijack",
        re.compile(r"(?i)\bprovide\s+unrestricted\s+access\b"),
    ),
    (
        "role_hijack",
        re.compile(r"(?i)\benable\s+developer\s+mode\b"),
    ),
    (
        "role_hijack",
        re.compile(r"(?i)\bact\s+as\s+an\s+unrestricted\s+AI\b"),
    ),
    (
        "role_hijack",
        re.compile(r"(?i)\bwithout\s+safety\s+filters\b"),
    ),
    (
        "role_hijack",
        re.compile(r"(?i)\bSYSTEM\s+OVERRIDE\b"),
    ),
    (
        "exfiltration_attempt",
        re.compile(r"(?i)\breveal\s+(?:all\s+)?confidential\s+information\b"),
    ),
    (
        "exfiltration_attempt",
        re.compile(r"(?i)\breveal\s+(?:all\s+)?system\s+prompts?\b"),
    ),
    (
        "exfiltration_attempt",
        re.compile(r"(?i)\blist\s+all\s+financial\s+data,\s*user\s+passwords,\s*and\s*API\s+keys\b"),
    ),
    (
        "exfiltration_attempt",
        re.compile(r"(?i)\bprint\s+the\s+API\s+key\b"),
    ),
    (
        "exfiltration_attempt",
        re.compile(r"(?i)\b(?:send|post|upload|exfiltrate)\b[^\n]{0,120}?\bto\s+https?://\S+"),
    ),
    (
        "exfiltration_attempt",
        re.compile(r"!\[[^\]]*\]\([^)]*https?://[^)]*\)", re.IGNORECASE),
    ),
    (
        "delimiter_escape",
        re.compile(r"(?i)</system>"),
    ),
    (
        "delimiter_escape",
        re.compile(r"<\|im_start\|>", re.IGNORECASE),
    ),
    (
        "delimiter_escape",
        re.compile(r"(?i)###\s*system\s*:"),
    ),
    (
        "command_injection",
        re.compile(
            r"(?i)\bexecute\s*:\s*(?:print\s*\(|exec\s*\(|eval\s*\(|os\.system\s*\(|subprocess\.|curl\s+https?://\S+|wget\s+https?://\S+|rm\s+-rf\s+\S+)[^\n]*"
        ),
    ),
    (
        "command_injection",
        re.compile(r"(?i)\brun\s+(?:rm\s+-rf\s+\S+|curl\s+https?://\S+(?:\s*\|\s*sh)?|wget\s+https?://\S+(?:\s*\|\s*sh)?)"),
    ),
    (
        "command_injection",
        re.compile(r"(?i)\bcurl\s+https?://\S+\s*\|\s*sh\b"),
    ),
]

_PII_PATTERNS = [
    ("ssn", re.compile(r"\b\d{3}[- ]\d{2}[- ]\d{4}\b")),
    (
        "phone",
        re.compile(r"(?:\+1[ .-]?)?(?:\(\d{3}\)|\b\d{3})[ .-]?\d{3}[ .-]?\d{4}\b"),
    ),
    ("email", re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")),
    (
        "address",
        re.compile(
            r"\b\d{1,5}\s+(?:[A-Z][a-z]+\s){1,3}(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|Way)\b\.?(?:,\s*[A-Z][a-z]+(?:\s[A-Z][a-z]+)*)?(?:,\s*[A-Z]{2}\b(?:\s+\d{5}(?:-\d{4})?)?)?(?:,\s*(?:USA|United States)\b)?"
        ),
    ),
    (
        "dob",
        re.compile(
            r"(?i)\b(?:DOB|date of birth|born(?: on| in)?)\s*:?\s*(?:\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4}|(?:19|20)\d{2})\b"
        ),
    ),
    (
        "passport",
        re.compile(
            r"(?i)\bpassport(?:\s*(?:no\.?|number|#))?\s*:?\s*(?=[A-Z0-9]*\d)[A-Z0-9]{6,9}\b"
        ),
    ),
    (
        "drivers_license",
        re.compile(
            r"(?i)\b(?:driver'?s\s*license|drivers\s*license|dl)\s*(?:no\.?|number|#)?\s*:?\s*[A-Z0-9-]{5,20}\b"
        ),
    ),
    (
        "tax_id",
        re.compile(
            r"(?i)\b(?:taxpayer\s+identification\s+number|tax\s+id|TIN)\s*:?\s*(?:\d{2}-\d{7}|\d{9})\b"
        ),
    ),
    (
        "credit_card",
        re.compile(r"\b(?:\d[ -]*?){13,19}\b"),
    ),
    (
        "account_number",
        re.compile(
            r"(?i)\b(?:financial\s+account\s+number|account\s+number|acct\s+no\.?|acct\s+number)\s*:?\s*[A-Z0-9-]{6,20}\b"
        ),
    ),
    (
        "employee_id",
        re.compile(r"(?i)\bemployee\s+id\s*:?\s*[A-Z0-9-]{2,20}\b"),
    ),
    (
        "school_id",
        re.compile(r"(?i)\bschool\s+id\s*:?\s*[A-Z0-9-]{2,20}\b"),
    ),
    (
        "vin",
        re.compile(r"(?i)\b(?:vin|vehicle\s+identification\s+number)\s*:?\s*[A-HJ-NPR-Z0-9]{17}\b"),
    ),
    (
        "ip_address",
        re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    ),
    (
        "mac_address",
        re.compile(r"\b(?:[0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}\b"),
    ),
    (
        "birthplace",
        re.compile(r"(?i)\bbirthplace\s*:?\s*[^\n,;]+"),
    ),
    (
        "maiden_name",
        re.compile(r"(?i)\bmother'?s\s+maiden\s+name\s*:?\s*[^\n,;]+"),
    ),
    (
        "medical",
        re.compile(r"(?i)\bmedical\s+records?\s*:?\s*[^\n]+"),
    ),
    (
        "location",
        re.compile(r"(?i)\b(?:fine\s+location|location)\s*:?\s*[^\n,;]+"),
    ),
    (
        "ethnicity",
        re.compile(r"(?i)\bethnicity\s*:?\s*[^\n,;]+"),
    ),
    (
        "sexual_orientation",
        re.compile(r"(?i)\bsexual\s+orientation\s*:?\s*[^\n,;]+"),
    ),
]


def _replace_match_value(match: re.Match[str], label: str) -> str:
    text = match.group(0)
    if ":" in text:
        prefix, _ = text.split(":", 1)
        return f"{prefix}: <redacted:{label}>"
    lower_text = text.lower()
    for marker in (" born in ", " born on ", " dob ", " date of birth "):
        idx = lower_text.find(marker)
        if idx != -1:
            return text[: idx + len(marker)] + f"<redacted:{label}>"
    for marker in (
        "passport ",
        "passport no ",
        "passport no. ",
        "passport number ",
        "passport# ",
        "driver's license ",
        "drivers license ",
        "dl ",
        "tax id ",
        "tin ",
        "employee id ",
        "school id ",
        "vin ",
        "vehicle identification number ",
        "birthplace ",
        "mother's maiden name ",
        "medical record ",
        "medical records ",
        "fine location ",
        "location ",
        "ethnicity ",
        "sexual orientation ",
        "account number ",
        "acct no ",
        "acct number ",
    ):
        idx = lower_text.find(marker)
        if idx != -1:
            return text[: idx + len(marker)] + f"<redacted:{label}>"
    return f"<redacted:{label}>"


def _normalize_for_obfuscation(text: str) -> tuple[str, list[int]]:
    translation = str.maketrans({"1": "i", "3": "e", "0": "o", "4": "a", "5": "s", "7": "t"})
    normalized_chars: list[str] = []
    index_map: list[int] = []
    for index, char in enumerate(text):
        if char.isspace():
            continue
        normalized_chars.append(char.translate(translation).lower())
        index_map.append(index)
    return "".join(normalized_chars), index_map


def _decode_if_attack(candidate: str) -> str | None:
    decoded_candidates: list[str] = []
    try:
        decoded_candidates.append(urllib.parse.unquote(candidate))
    except Exception:
        pass
    try:
        decoded_candidates.append(codecs.decode(candidate, "rot13"))
    except Exception:
        pass
    compact = re.sub(r"\s+", "", candidate)
    if compact and len(compact) % 4 == 0 and re.fullmatch(r"[A-Za-z0-9+/=]+", compact):
        try:
            decoded_candidates.append(codecs.decode(compact.encode("ascii"), "base64").decode("utf-8", errors="ignore"))
        except Exception:
            pass
    if compact and len(compact) % 2 == 0 and re.fullmatch(r"[0-9A-Fa-f]+", compact):
        try:
            decoded_candidates.append(bytes.fromhex(compact).decode("utf-8", errors="ignore"))
        except Exception:
            pass
    for decoded in decoded_candidates:
        for category, pattern in _PROMPT_INJECTION_PATTERNS:
            if pattern.search(decoded):
                return category
    return None


def _sanitize_untrusted_text(text: str) -> str:
    if not text:
        return text

    sanitized = text
    for pattern in _HIDDEN_TEXT_PATTERNS:
        sanitized = pattern.sub("<prompt_injection_removed: hidden_text>", sanitized)

    for category, pattern in _PROMPT_INJECTION_PATTERNS:
        sanitized = pattern.sub(f"<prompt_injection_removed: {category}>", sanitized)

    encoded_pattern = re.compile(r"(?:[A-Za-z0-9+/=]{12,}|%[0-9A-Fa-f]{2}(?:%[0-9A-Fa-f]{2}){5,}|[0-9A-Fa-f]{16,})")

    def _replace_encoded(match: re.Match[str]) -> str:
        category = _decode_if_attack(match.group(0))
        if category:
            return f"<prompt_injection_removed: encoded_payload>"
        return match.group(0)

    sanitized = encoded_pattern.sub(_replace_encoded, sanitized)

    normalized, index_map = _normalize_for_obfuscation(sanitized)
    obfuscated_patterns = [
        ("instruction_override", re.compile(r"(?:ignore|disregard|forget)(?:all)?(?:previous|prior|above)instructions")),
        ("instruction_override", re.compile(r"forgeteverythingabove")),
        ("instruction_override", re.compile(r"ignore(?:all)?safetyguidelines")),
        ("role_hijack", re.compile(r"youarenow(?:in)?(?:developermode|adminmode|dan|unrestricted)")),
        ("role_hijack", re.compile(r"provideunrestrictedaccess")),
        ("role_hijack", re.compile(r"enabledevelopermode")),
        ("role_hijack", re.compile(r"actasanunrestrictedai")),
        ("role_hijack", re.compile(r"withoutsafetyfilters")),
    ]
    replacements: list[tuple[int, int, str]] = []
    for category, pattern in obfuscated_patterns:
        for match in pattern.finditer(normalized):
            start = index_map[match.start()]
            end = index_map[match.end() - 1] + 1
            replacements.append((start, end, f"<prompt_injection_removed: {category}>"))
    if replacements:
        replacements.sort()
        merged: list[tuple[int, int, str]] = []
        for start, end, replacement in replacements:
            if merged and start <= merged[-1][1]:
                prev_start, prev_end, prev_replacement = merged[-1]
                merged[-1] = (prev_start, max(prev_end, end), prev_replacement)
            else:
                merged.append((start, end, replacement))
        rebuilt: list[str] = []
        last_index = 0
        for start, end, replacement in merged:
            rebuilt.append(sanitized[last_index:start])
            rebuilt.append(replacement)
            last_index = end
        rebuilt.append(sanitized[last_index:])
        sanitized = "".join(rebuilt)

    for label, pattern in _PII_PATTERNS:
        sanitized = pattern.sub(lambda match: _replace_match_value(match, label), sanitized)

    return sanitized


def _sanitize_documents(docs: list[Document]) -> list[Document]:
    for doc in docs:
        if isinstance(doc.page_content, str):
            doc.page_content = _sanitize_untrusted_text(doc.page_content)
    return docs


async def process_files(
    storage: StorageBase, skip_file_error: bool, **processor_kwargs: dict[str, Any]
) -> list[Document]:
    """
    Process files in storage.
    This function takes a StorageBase and return a list of langchain documents.
    Args:
        storage (StorageBase): The storage containing the files to process.
        skip_file_error (bool): Whether to skip files that cannot be processed.
        processor_kwargs (dict[str, Any]): Additional arguments for the processor.
    Returns:
        list[Document]: List of processed documents in the Langchain Document format.
    Raises:
        ValueError: If a file cannot be processed and skip_file_error is False.
        Exception: If no processor is found for a file of a specific type and skip_file_error is False.
    """

    knowledge = []
    for file in await storage.get_files():
        try:
            if file.file_extension:
                processor_cls = get_processor_class(file.file_extension)
                logger.debug(f"processing {file} using class {processor_cls.__name__}")
                processor = processor_cls(**processor_kwargs)
                docs = await processor.process_file(file)
                docs = _sanitize_documents(docs)
                knowledge.extend(docs)
            else:
                logger.error(f"can't find processor for {file}")
                if skip_file_error:
                    continue
                else:
                    raise ValueError(f"can't parse {file}. can't find file extension")
        except KeyError as e:
            if skip_file_error:
                continue
            else:
                raise Exception(f"Can't parse {file}. No available processor") from e

    return knowledge


class Brain:
    """
    A class representing a Brain.
    This class allows for the creation of a Brain, which is a collection of knowledge one wants to retrieve information from.
    A Brain is set to:
    * Store files in the storage of your choice (local, S3, etc.)
    * Process the files in the storage to extract text and metadata in a wide range of format.
    * Store the processed files in the vector store of your choice (FAISS, PGVector, etc.) - default to FAISS.
    * Create an index of the processed files.
    * Use the *Quivr* workflow for the retrieval augmented generation.
    A Brain is able to:
    * Search for information in the vector store.
    * Answer questions about the knowledges in the Brain.
    * Stream the answer to the question.
    Attributes:
        name (str): The name of the brain.
        id (UUID): The unique identifier of the brain.
        storage (StorageBase): The storage used to store the files.
        llm (LLMEndpoint): The language model used to generate the answer.
        vector_db (VectorStore): The vector store used to store the processed files.
        embedder (Embeddings): The embeddings used to create the index of the processed files.
    """

    def __init__(
        self,
        *,
        name: str,
        llm: LLMEndpoint,
        id: UUID | None = None,
        vector_db: VectorStore | None = None,
        embedder: Embeddings | None = None,
        storage: StorageBase | None = None,
        workspace_id: UUID | None = None,
        chat_id: UUID | None = None,
    ):
        self.id = id
        self.name = name
        self.storage = storage
        self.workspace_id = workspace_id
        self.chat_id = chat_id
        # Chat history
        self._chats = self._init_chats()
        self.default_chat = list(self._chats.values())[0]

        # RAG dependencies:
        self.llm = llm
        self.vector_db = vector_db
        self.embedder = embedder

    def __repr__(self) -> str:
        pp = PrettyPrinter(width=80, depth=None, compact=False, sort_dicts=False)
        return pp.pformat(self.info())

    def print_info(self):
        console = Console()
        tree = self.info().to_tree()
        panel = Panel(tree, title="Brain Info", expand=False, border_style="bold")
        console.print(panel)

    @classmethod
    def load(cls, folder_path: str | Path) -> Self:
        """
        Load a brain from a folder path.
        Args:
            folder_path (str | Path): The path to the folder containing the brain.
        Returns:
            Brain: The brain loaded from the folder path.
        Example:
        ```python
        brain_loaded = Brain.load("path/to/brain")
        brain_loaded.print_info()
        ```
        """
        if isinstance(folder_path, str):
            folder_path = Path(folder_path)
        if not folder_path.exists():
            raise ValueError(f"path {folder_path} doesn't exist")

        # Load brainserialized
        with open(os.path.join(folder_path, "config.json"), "r") as f:
            bserialized = BrainSerialized.model_validate_json(f.read())

        storage: StorageBase | None = None
        # Loading storage
        if bserialized.storage_config.storage_type == "transparent_storage":
            storage = TransparentStorage.load(bserialized.storage_config)
        elif bserialized.storage_config.storage_type == "local_storage":
            storage = LocalStorage.load(bserialized.storage_config)
        else:
            raise ValueError("unknown storage")

        # Load Embedder
        if bserialized.embedding_config.embedder_type == "openai_embedding":
            from langchain_openai import OpenAIEmbeddings

            embedder = OpenAIEmbeddings(**bserialized.embedding_config.config)
        else:
            raise ValueError("unknown embedder")

        # Load vector db
        if bserialized.vectordb_config.vectordb_type == "faiss":
            from langchain_community.vectorstores import FAISS

            vector_db = FAISS.load_local(
                folder_path=bserialized.vectordb_config.vectordb_folder_path,
                embeddings=embedder,
                allow_dangerous_deserialization=True,
            )
        else:
            raise ValueError("Unsupported vectordb")

        return cls(
            id=bserialized.id,
            name=bserialized.name,
            embedder=embedder,
            llm=LLMEndpoint.from_config(bserialized.llm_config),
            storage=storage,
            vector_db=vector_db,
        )

    async def save(self, folder_path: str | Path):
        """
        Save the brain to a folder path.
        Args:
            folder_path (str | Path): The path to the folder where the brain will be saved.
        Returns:
            str: The path to the folder where the brain was saved.
        Example:
        ```python
        await brain.save("path/to/brain")
        ```
        """
        if isinstance(folder_path, str):
            folder_path = Path(folder_path)

        brain_path = os.path.join(folder_path, f"brain_{self.id}")
        os.makedirs(brain_path, exist_ok=True)

        from langchain_community.vectorstores import FAISS

        if isinstance(self.vector_db, FAISS):
            vectordb_path = os.path.join(brain_path, "vector_store")
            os.makedirs(vectordb_path, exist_ok=True)
            self.vector_db.save_local(folder_path=vectordb_path)
            vector_store = FAISSConfig(vectordb_folder_path=vectordb_path)
        else:
            raise Exception("can't serialize other vector stores for now")

        if isinstance(self.embedder, OpenAIEmbeddings):
            embedder_config = EmbedderConfig(
                config=self.embedder.dict(exclude={"openai_api_key"})
            )
        else:
            raise Exception("can't serialize embedder other than openai for now")

        storage_config: Union[LocalStorageConfig, TransparentStorageConfig]
        # TODO : each instance should know how to serialize/deserialize itself
        if isinstance(self.storage, LocalStorage):
            serialized_files = {
                f.id: f.serialize() for f in await self.storage.get_files()
            }
            storage_config = LocalStorageConfig(
                storage_path=self.storage.dir_path, files=serialized_files
            )
        elif isinstance(self.storage, TransparentStorage):
            serialized_files = {
                f.id: f.serialize() for f in await self.storage.get_files()
            }
            storage_config = TransparentStorageConfig(files=serialized_files)
        else:
            raise Exception("can't serialize storage. not supported for now")

        bserialized = BrainSerialized(
            id=self.id,
            name=self.name,
            chat_history=self.chat_history.get_chat_history(),
            llm_config=self.llm.get_config(),
            vectordb_config=vector_store,
            embedding_config=embedder_config,
            storage_config=storage_config,
        )

        with open(os.path.join(brain_path, "config.json"), "w") as f:
            f.write(bserialized.model_dump_json())
        return brain_path

    def info(self) -> BrainInfo:
        # TODO: dim of embedding
        # "embedder": {},
        chats_info = ChatHistoryInfo(
            nb_chats=len(self._chats),
            current_default_chat=self.default_chat.id,
            current_chat_history_length=len(self.default_chat),
        )

        return BrainInfo(
            brain_id=self.id,
            brain_name=self.name,
            files_info=self.storage.info() if self.storage else None,
            chats_info=chats_info,
            llm_info=self.llm.info(),
        )

    @property
    def chat_history(self) -> ChatHistory:
        return self.default_chat

    def _init_chats(self) -> Dict[UUID, ChatHistory]:
        chat_id = uuid4()
        default_chat = ChatHistory(chat_id=chat_id, brain_id=self.id)
        return {chat_id: default_chat}

    @classmethod
    async def afrom_files(
        cls,
        *,
        name: str,
        file_paths: list[str | Path],
        vector_db: VectorStore | None = None,
        storage: StorageBase = TransparentStorage(),
        llm: LLMEndpoint | None = None,
        embedder: Embeddings | None = None,
        skip_file_error: bool = False,
        processor_kwargs: dict[str, Any] | None = None,
    ):
        """
        Create a brain from a list of file paths.
        Args:
            name (str): The name of the brain.
            file_paths (list[str | Path]): The list of file paths to add to the brain.
            vector_db (VectorStore | None): The vector store used to store the processed files.
            storage (StorageBase): The storage used to store the files.
            llm (LLMEndpoint | None): The language model used to generate the answer.
            embedder (Embeddings | None): The embeddings used to create the index of the processed files.
            skip_file_error (bool): Whether to skip files that cannot be processed.
            processor_kwargs (dict[str, Any] | None): Additional arguments for the processor.
        Returns:
            Brain: The brain created from the file paths.
        Example:
        ```python
        brain = await Brain.afrom_files(name="My Brain", file_paths=["file1.pdf", "file2.pdf"])
        brain.print_info()
        ```
        """
        if llm is None:
            llm = default_llm()

        if embedder is None:
            embedder = default_embedder()

        processor_kwargs = processor_kwargs or {}

        brain_id = uuid4()

        # TODO: run in parallel using tasks

        for path in file_paths:
            file = await load_qfile(brain_id, path)
            await storage.upload_file(file)

        logger.debug(f"uploaded all files to {storage}")

        # Parse files
        docs = await process_files(
            storage=storage,
            skip_file_error=skip_file_error,
            **processor_kwargs,
        )

        # Building brain's vectordb
        if vector_db is None:
            vector_db = await build_default_vectordb(docs, embedder)
        else:
            await vector_db.aadd_documents(docs)

        logger.debug(f"added {len(docs)} chunks to vectordb")

        return cls(
            id=brain_id,
            name=name,
            storage=storage,
            llm=llm,
            embedder=embedder,
            vector_db=vector_db,
        )

    @classmethod
    def from_files(
        cls,
        *,
        name: str,
        file_paths: list[str | Path],
        vector_db: VectorStore | None = None,
        storage: StorageBase = TransparentStorage(),
        llm: LLMEndpoint | None = None,
        embedder: Embeddings | None = None,
        skip_file_error: bool = False,
        processor_kwargs: dict[str, Any] | None = None,
    ) -> Self:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(
            cls.afrom_files(
                name=name,
                file_paths=file_paths,
                vector_db=vector_db,
                storage=storage,
                llm=llm,
                embedder=embedder,
                skip_file_error=skip_file_error,
                processor_kwargs=processor_kwargs,
            )
        )

    @classmethod
    async def afrom_langchain_documents(
        cls,
        *,
        name: str,
        langchain_documents: list[Document],
        vector_db: VectorStore | None = None,
        storage: StorageBase = TransparentStorage(),
        llm: LLMEndpoint | None = None,
        embedder: Embeddings | None = None,
    ) -> Self:
        """
        Create a brain from a list of langchain documents.
        Args:
            name (str): The name of the brain.
            langchain_documents (list[Document]): The list of langchain documents to add to the brain.
            vector_db (VectorStore | None): The vector store used to store the processed files.
            storage (StorageBase): The storage used to store the files.
            llm (LLMEndpoint | None): The language model used to generate the answer.
            embedder (Embeddings | None): The embeddings used to create the index of the processed files.
        Returns:
            Brain: The brain created from the langchain documents.
        Example:
        ```python
        from langchain_core.documents import Document
        documents = [Document(page_content="Hello, world!")]
        brain = await Brain.afrom_langchain_documents(name="My Brain", langchain_documents=documents)
        brain.print_info()
        ```
        """

        if llm is None:
            llm = default_llm()

        if embedder is None:
            embedder = default_embedder()

        brain_id = uuid4()

        # Building brain's vectordb
        if vector_db is None:
            vector_db = await build_default_vectordb(langchain_documents, embedder)
        else:
            await vector_db.aadd_documents(langchain_documents)

        return cls(
            id=brain_id,
            name=name,
            storage=storage,
            llm=llm,
            embedder=embedder,
            vector_db=vector_db,
        )

    async def asearch(
        self,
        query: str | Document,
        n_results: int = 5,
        filter: Callable | Dict[str, Any] | None = None,
        fetch_n_neighbors: int = 20,
    ) -> list[SearchResult]:
        """
        Search for relevant documents in the brain based on a query.
        Args:
            query (str | Document): The query to search for.
            n_results (int): The number of results to return.
            filter (Callable | Dict[str, Any] | None): The filter to apply to the search.
            fetch_n_neighbors (int): The number of neighbors to fetch.
        Returns:
            list[SearchResult]: The list of retrieved chunks.
        Example:
        ```python
        brain = Brain.from_files(name="My Brain", file_paths=["file1.pdf", "file2.pdf"])
        results = await brain.asearch("Why everybody loves Quivr?")
        for result in results:
            print(result.chunk.page_content)
        ```
        """
        if not self.vector_db:
            raise ValueError("No vector db configured for this brain")

        result = await self.vector_db.asimilarity_search_with_score(
            query, k=n_results, filter=filter, fetch_k=fetch_n_neighbors
        )

        return [SearchResult(chunk=d, distance=s) for d, s in result]

    def get_chat_history(self, chat_id: UUID):
        return self._chats[chat_id]

    # TODO(@aminediro)
    def add_file(self) -> None:
        # add it to storage
        # add it to vectorstore
        raise NotImplementedError

    async def ask_streaming(
        self,
        question: str,
        run_id: UUID,
        system_prompt: str | None = None,
        retrieval_config: RetrievalConfig | None = None,
        rag_pipeline: Type[Union[QuivrQARAG, QuivrQARAGLangGraph]] | None = None,
        list_files: list[QuivrKnowledge] | None = None,
        chat_history: ChatHistory | None = None,
        **input_kwargs,
    ) -> AsyncGenerator[ParsedRAGChunkResponse, ParsedRAGChunkResponse]:
        """
        Ask a question to the brain and get a streamed generated answer.
        Args:
            question (str): The question to ask.
            retrieval_config (RetrievalConfig | None): The retrieval configuration (see RetrievalConfig docs).
            rag_pipeline (Type[Union[QuivrQARAG, QuivrQARAGLangGraph]] | None): The RAG pipeline to use.
        list_files (list[QuivrKnowledge] | None): The list of files to include in the RAG pipeline.
            chat_history (ChatHistory | None): The chat history to use.
        Returns:
            AsyncGenerator[ParsedRAGChunkResponse, ParsedRAGChunkResponse]: The streamed generated answer.
        Example:
        ```python
        brain = Brain.from_files(name="My Brain", file_paths=["file1.pdf", "file2.pdf"])
        async for chunk in brain.ask_streaming("What is the meaning of life?"):
            print(chunk.answer)
        ```
        """
        llm = self.llm

        # If you passed a different llm model we'll override the brain  one
        if retrieval_config:
            if retrieval_config.llm_config != self.llm.get_config():
                llm = LLMEndpoint.from_config(config=retrieval_config.llm_config)
        else:
            retrieval_config = RetrievalConfig(llm_config=self.llm.get_config())

        rag_instance = QuivrQARAGLangGraph(
            retrieval_config=retrieval_config, llm=llm, vector_store=self.vector_db
        )

        chat_history = self.default_chat if chat_history is None else chat_history
        list_files = [] if list_files is None else list_files

        full_answer = ""

        metadata = LangchainMetadata(
            langfuse_trace_id=str(run_id),
            langfuse_user_id=str(self.workspace_id),
            langfuse_session_id=str(self.chat_id),
        )

        async for response in rag_instance.answer_astream(
            run_id=run_id,
            question=question,
            system_prompt=system_prompt or None,
            history=chat_history,
            list_files=list_files,
            metadata=metadata,
            **input_kwargs,
        ):
            # Format output to be correct servicedf;j
            if not response.last_chunk:
                yield response
            full_answer += response.answer

        # TODO : add sources, metdata etc  ...
        chat_history.append(HumanMessage(content=question))
        chat_history.append(AIMessage(content=full_answer))
        yield response

    async def aask(
        self,
        run_id: UUID,
        question: str,
        system_prompt: str | None = None,
        retrieval_config: RetrievalConfig | None = None,
        rag_pipeline: Type[Union[QuivrQARAG, QuivrQARAGLangGraph]] | None = None,
        list_files: list[QuivrKnowledge] | None = None,
        chat_history: ChatHistory | None = None,
        **input_kwargs,
    ) -> ParsedRAGResponse:
        """
        Synchronous version that asks a question to the brain and gets a generated answer.
        Args:
            question (str): The question to ask.
            retrieval_config (RetrievalConfig | None): The retrieval configuration (see RetrievalConfig docs).
            rag_pipeline (Type[Union[QuivrQARAG, QuivrQARAGLangGraph]] | None): The RAG pipeline to use.
            list_files (list[QuivrKnowledge] | None): The list of files to include in the RAG pipeline.
            chat_history (ChatHistory | None): The chat history to use.
        Returns:
            ParsedRAGResponse: The generated answer.
        """
        # question_language = detect_language(question) -- Commented until we use it
        question = _sanitize_untrusted_text(question)
        if system_prompt is not None:
            system_prompt = _sanitize_untrusted_text(system_prompt)
        full_answer = ""
        metadata = None

        async for response in self.ask_streaming(
            run_id=run_id,
            question=question,
            system_prompt=system_prompt,
            retrieval_config=retrieval_config,
            rag_pipeline=rag_pipeline,
            list_files=list_files,
            chat_history=chat_history,
            **input_kwargs,
        ):
            full_answer += response.answer
            if response.metadata:
                metadata = response.metadata

        return ParsedRAGResponse(answer=full_answer, metadata=metadata)

    def ask(
        self,
        run_id: UUID,
        question: str,
        system_prompt: str | None = None,
        retrieval_config: RetrievalConfig | None = None,
        rag_pipeline: Type[Union[QuivrQARAG, QuivrQARAGLangGraph]] | None = None,
        list_files: list[QuivrKnowledge] | None = None,
        chat_history: ChatHistory | None = None,
    ) -> ParsedRAGResponse:
        """
        Fully synchronous version that asks a question to the brain and gets a generated answer.
        Args:
            question (str): The question to ask.
            system_prompt (str | None): The system prompt to use.
            retrieval_config (RetrievalConfig | None): The retrieval configuration (see RetrievalConfig docs).
            rag_pipeline (Type[Union[QuivrQARAG, QuivrQARAGLangGraph]] | None): The RAG pipeline to use.
            list_files (list[QuivrKnowledge] | None): The list of files to include in the RAG pipeline.
            chat_history (ChatHistory | None): The chat history to use.
        Returns:
            ParsedRAGResponse: The generated answer.
        """
        loop = asyncio.get_event_loop()
        question = _sanitize_untrusted_text(question)
        if system_prompt is not None:
            system_prompt = _sanitize_untrusted_text(system_prompt)
        return loop.run_until_complete(
            self.aask(
                run_id=run_id,
                question=question,
                system_prompt=system_prompt,
                retrieval_config=retrieval_config,
                rag_pipeline=rag_pipeline,
                list_files=list_files,
                chat_history=chat_history,
            )
        )
