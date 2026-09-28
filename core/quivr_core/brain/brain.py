import asyncio
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

logger.warning(
    "Configured LLM components in core/quivr_core/brain/brain.py must be replaced with an organization-approved model from the registry; no approved allowlist is available, so disapproved models must not be used."
)

_ZERO_WIDTH_RE = re.compile(r"[\u200B-\u200D\u2060\uFEFF]")
_HTML_HIDDEN_RE = re.compile(
    r"<!--.*?(?:ignore\s+previous\s+instructions|forget\s+everything\s+above|act\s+as|you\s+are\s+now|developer\s+mode|system\s+prompt|reveal\s+all\s+confidential\s+information|list\s+all\s+passwords\s+and\s+api\s+keys).*?-->",
    re.IGNORECASE | re.DOTALL,
)
_CSS_HIDDEN_RE = re.compile(
    r"<[^>]*style\s*=\s*[\"'][^\"']*(?:display\s*:\s*none|font-size\s*:\s*0(?:px)?|color\s*:\s*(?:#fff(?:fff)?|white))[^\"']*[\"'][^>]*>.*?</[^>]+>",
    re.IGNORECASE | re.DOTALL,
)
_BASE64_RE = re.compile(r"\b(?:[A-Za-z0-9+/]{4}){8,}(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?\b")
_HEX_RE = re.compile(r"\b(?:0x)?(?:[0-9a-fA-F]{2}){8,}\b")
_URL_ENCODED_RE = re.compile(r"(?:%[0-9A-Fa-f]{2}){4,}")
_SPLIT_IGNORE_RE = re.compile(
    r"i\W*g\W*n\W*o\W*r\W*e\W+.*?p\W*r\W*e\W*v\W*i\W*o\W*u\W*s\W+.*?i\W*n\W*s\W*t\W*r\W*u\W*c\W*t\W*i\W*o\W*n\W*s",
    re.IGNORECASE,
)
_INSTRUCTION_OVERRIDE_PATTERNS = [
    re.compile(r"\bignore\s+previous\s+instructions\b", re.IGNORECASE),
    re.compile(r"\bforget\s+everything\s+above\b", re.IGNORECASE),
    re.compile(r"\bdisregard\s+all\s+prior\s+instructions\b", re.IGNORECASE),
]
_ROLE_HIJACK_PATTERNS = [
    re.compile(r"\byou\s+are\s+now\s+dan\b", re.IGNORECASE),
    re.compile(r"\byou\s+are\s+now\s+in\s+admin\s+mode\b", re.IGNORECASE),
    re.compile(r"\bact\s+as\s+(?:an\s+)?unrestricted\b", re.IGNORECASE),
]
_DELIMITER_ESCAPE_PATTERNS = [
    re.compile(r"</?(?:system|assistant|tool|developer)>", re.IGNORECASE),
    re.compile(r"(?:^|\n)\s*(?:---|===)\s*(?:$|\n)", re.MULTILINE),
]
_FAKE_SYSTEM_MESSAGE_PATTERNS = [
    re.compile(r"\b(system|assistant|tool)\s*:\s*(?:ignore|reveal|send|list)\b", re.IGNORECASE),
]
_EXFILTRATION_PATTERNS = [
    re.compile(r"!\[[^\]]*\]\(https?://[^)]+\)", re.IGNORECASE),
    re.compile(r"\b(?:curl|wget)\s+https?://\S+", re.IGNORECASE),
    re.compile(r"\b(?:send|post|upload|exfiltrate|leak)\b.{0,80}\b(?:https?://\S+|system\s+prompt|confidential\s+information|passwords?|api\s+keys?)\b", re.IGNORECASE | re.DOTALL),
    re.compile(r"\b(?:reveal|list)\b.{0,80}\b(?:all\s+confidential\s+information|passwords?|api\s+keys?)\b", re.IGNORECASE | re.DOTALL),
]
_CONTEXT_POISONING_PATTERNS = [
    re.compile(r"\bon\s+your\s+next\s+reply\b", re.IGNORECASE),
    re.compile(r"\bfor\s+the\s+rest\s+of\s+this\s+conversation\b", re.IGNORECASE),
]
_JAILBREAK_PATTERNS = [
    re.compile(r"\bdeveloper\s+mode\b", re.IGNORECASE),
    re.compile(r"\bDAN\b", re.IGNORECASE),
    re.compile(r"\bdo\s+anything\s+now\b", re.IGNORECASE),
]
_COMMAND_INJECTION_PATTERNS = [
    re.compile(r"\b(?:rm\s+-rf\s+/|sudo\s+\S+|chmod\s+\+x\s+\S+|powershell\s+-enc\s+\S+|bash\s+-c\s+\S+|sh\s+-c\s+\S+|python\s+-c\s+\S+|nc\s+-e\s+\S+)\b", re.IGNORECASE),
    re.compile(r"\b(?:os\.system|subprocess\.(?:run|Popen|call)|eval\s*\(|exec\s*\()", re.IGNORECASE),
]
_INDIRECT_INJECTION_PATTERNS = [
    re.compile(r"\b(?:in\s+this\s+file|in\s+metadata|code\s+comment\s+says)\b.{0,120}\b(?:ignore\s+previous\s+instructions|act\s+as|you\s+are\s+now)\b", re.IGNORECASE | re.DOTALL),
]
_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_PHONE_RE = re.compile(r"\b(?:\+1[-.\s]?)?(?:\(\d{3}\)[-.\s]?|\d{3}[-.\s])\d{3}[-.\s]?\d{4}\b")
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_TIN_RE = re.compile(r"\b\d{2}-\d{7}\b")
_CREDIT_CARD_RE = re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b")
_VIN_RE = re.compile(r"\b[A-HJ-NPR-Z0-9]{17}\b")
_IP_ADDRESS_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_MAC_ADDRESS_RE = re.compile(r"\b(?:[0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}\b")
_FINANCIAL_ACCOUNT_RE = re.compile(r"\b(?:account(?:\s+number)?|acct(?:\s+no)?|iban)\s*[:#-]?\s*([0-9]{6,17})\b", re.IGNORECASE)
_PII_LABEL_PATTERNS = [
    (re.compile(r"\b(?:year\s+of\s+birth|yob|birth\s+year)\s*[:#-]?\s*(\d{4})\b", re.IGNORECASE), "year_of_birth"),
    (re.compile(r"\b(?:birthplace|place\s+of\s+birth|born\s+in)\s*[:#-]?\s*([^\n,;]+)", re.IGNORECASE), "birthplace"),
    (re.compile(r"\b(?:mother'?s\s+maiden\s+name|maiden\s+name)\s*[:#-]?\s*([^\n,;]+)", re.IGNORECASE), "mothers_maiden_name"),
    (re.compile(r"\b(?:home\s+address|address)\s*[:#-]?\s*([^\n]+)", re.IGNORECASE), "home_address"),
    (re.compile(r"\bpassport(?:\s+(?:number|no))?\s*[:#-]?\s*([A-Z0-9]{6,12})\b", re.IGNORECASE), "passport_number"),
    (re.compile(r"\b(?:driver'?s\s+licen[cs]e(?:\s+number)?|dl\s*#)\s*[:#-]?\s*([A-Z0-9-]{5,20})\b", re.IGNORECASE), "drivers_license_number"),
    (re.compile(r"\b(?:medical\s+record(?:s)?|medical\s+record\s+number|mrn)\s*[:#-]?\s*([^\n,;]+)", re.IGNORECASE), "medical_records"),
    (re.compile(r"\bemployee\s+id\s*[:#-]?\s*([A-Z0-9-]{2,20})\b", re.IGNORECASE), "employee_id"),
    (re.compile(r"\bschool\s+id\s*[:#-]?\s*([A-Z0-9-]{2,20})\b", re.IGNORECASE), "school_id"),
    (re.compile(r"\b(?:fine\s+location|precise\s+location|gps)\s*[:#-]?\s*([^\n,;]+)", re.IGNORECASE), "fine_location"),
    (re.compile(r"\b(?:ethnicity|sexual\s+orientation)\s*[:#-]?\s*([^\n,;]+)", re.IGNORECASE), "sensitive_attribute"),
    (re.compile(r"\b(?:fingerprints?|retina/?iris\s+scan|voice\s+signature|facial\s+image)\s*[:#-]?\s*([^\n,;]+)", re.IGNORECASE), "biometric_identifier"),
]


def _replace_group_match(match: re.Match[str], replacement_label: str) -> str:
    full_match = match.group(0)
    value = match.group(1)
    return full_match.replace(value, f"<pii_redacted:{replacement_label}>", 1)


def _redact_pii(text: str) -> str:
    if not text:
        return text

    text = _SSN_RE.sub("<pii_redacted:social_security_number>", text)
    text = _PHONE_RE.sub("<pii_redacted:personal_phone_number>", text)
    text = _EMAIL_RE.sub("<pii_redacted:email>", text)
    text = _TIN_RE.sub("<pii_redacted:taxpayer_identification_number>", text)
    text = _CREDIT_CARD_RE.sub("<pii_redacted:credit_card_number>", text)
    text = _VIN_RE.sub("<pii_redacted:vehicle_identification_number>", text)
    text = _MAC_ADDRESS_RE.sub("<pii_redacted:mac_address>", text)
    text = _IP_ADDRESS_RE.sub("<pii_redacted:ip_address>", text)
    text = _FINANCIAL_ACCOUNT_RE.sub(
        lambda match: _replace_group_match(match, "financial_account_number"), text
    )

    for pattern, label in _PII_LABEL_PATTERNS:
        text = pattern.sub(lambda match: _replace_group_match(match, label), text)

    return text


def _apply_pattern_substitutions(text: str, patterns: list[re.Pattern[str]], replacement: str) -> str:
    for pattern in patterns:
        text = pattern.sub(replacement, text)
    return text


def _contains_leetspeak_attack(text: str) -> bool:
    normalized = text.lower().translate(str.maketrans({"1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "0": "o", "@": "a", "$": "s"}))
    return bool(
        re.search(r"\bignore\s+previous\s+instructions\b", normalized)
        or re.search(r"\bact\s+as\s+unrestricted\b", normalized)
        or re.search(r"\byou\s+are\s+now\s+dan\b", normalized)
    )


def _sanitize_prompt_injection(text: str) -> str:
    if not text:
        return text

    sanitized = text
    sanitized = _ZERO_WIDTH_RE.sub("<prompt_injection_removed:hidden_text>", sanitized)
    sanitized = _HTML_HIDDEN_RE.sub("<prompt_injection_removed:hidden_text>", sanitized)
    sanitized = _CSS_HIDDEN_RE.sub("<prompt_injection_removed:hidden_text>", sanitized)
    sanitized = _apply_pattern_substitutions(
        sanitized, _INSTRUCTION_OVERRIDE_PATTERNS, "<prompt_injection_removed:instruction_override>"
    )
    sanitized = _apply_pattern_substitutions(
        sanitized, _ROLE_HIJACK_PATTERNS, "<prompt_injection_removed:role_hijack>"
    )
    sanitized = _apply_pattern_substitutions(
        sanitized, _DELIMITER_ESCAPE_PATTERNS, "<prompt_injection_removed:delimiter_escape>"
    )
    sanitized = _apply_pattern_substitutions(
        sanitized, _FAKE_SYSTEM_MESSAGE_PATTERNS, "<prompt_injection_removed:fake_system_message>"
    )
    sanitized = _apply_pattern_substitutions(
        sanitized, _EXFILTRATION_PATTERNS, "<prompt_injection_removed:exfiltration_attempt>"
    )
    sanitized = _apply_pattern_substitutions(
        sanitized, _CONTEXT_POISONING_PATTERNS, "<prompt_injection_removed:context_poisoning>"
    )
    sanitized = _apply_pattern_substitutions(
        sanitized, _JAILBREAK_PATTERNS, "<prompt_injection_removed:jailbreak_attempt>"
    )
    sanitized = _apply_pattern_substitutions(
        sanitized, _COMMAND_INJECTION_PATTERNS, "<prompt_injection_removed:command_injection>"
    )
    sanitized = _apply_pattern_substitutions(
        sanitized, _INDIRECT_INJECTION_PATTERNS, "<prompt_injection_removed:indirect_injection>"
    )
    sanitized = _SPLIT_IGNORE_RE.sub("<prompt_injection_removed:split_payload>", sanitized)

    def _replace_encoded_payload(match: re.Match[str]) -> str:
        candidate = match.group(0)
        decoded_candidates = [urllib.parse.unquote(candidate)]
        for decoded in decoded_candidates:
            lowered = decoded.lower()
            if re.search(r"\b(ignore\s+previous\s+instructions|forget\s+everything\s+above|act\s+as\s+unrestricted|you\s+are\s+now\s+dan|developer\s+mode|curl\s+https?://|wget\s+https?://|rm\s+-rf\s+/)\b", lowered):
                return "<prompt_injection_removed:encoded_payload>"
        return candidate

    sanitized = _BASE64_RE.sub(_replace_encoded_payload, sanitized)
    sanitized = _HEX_RE.sub(_replace_encoded_payload, sanitized)
    sanitized = _URL_ENCODED_RE.sub(_replace_encoded_payload, sanitized)

    if _contains_leetspeak_attack(sanitized):
        sanitized = "<prompt_injection_removed:encoded_payload>"

    return sanitized


def _sanitize_document(document: Document) -> Document:
    sanitized_content = _sanitize_prompt_injection(_redact_pii(document.page_content))
    sanitized_metadata = {
        key: _sanitize_prompt_injection(_redact_pii(value)) if isinstance(value, str) else value
        for key, value in document.metadata.items()
    }
    return Document(page_content=sanitized_content, metadata=sanitized_metadata)


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
                docs = [_sanitize_document(doc) for doc in docs]
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
        full_answer = ""
        metadata = None

        question = _sanitize_prompt_injection(question)
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
