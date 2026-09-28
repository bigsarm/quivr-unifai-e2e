import logging
import os
import re
import time
from base64 import b64decode
from typing import Any, Sequence, Union
from urllib.parse import parse_qs, unquote, urlparse

import tiktoken
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_mistralai import ChatMistralAI
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from pydantic import SecretStr

from quivr_core.brain.info import LLMInfo
from quivr_core.rag.entities.config import DefaultModelSuppliers, LLMEndpointConfig
from quivr_core.rag.utils import model_supports_function_calling

logger = logging.getLogger("quivr_core")

_DISAPPROVED_MODEL_NAMES = {
    "deepseekchat",
    "deepseekr1",
    "deepseekr1distillllama70b",
    "deepseekreasoner",
    "customllmclientnull",
    "openrouternull",
    "usdeepseekr1v10null",
}


def _normalize_model_identifier(value: str) -> str:
    return re.sub(r"[\s\-_.:]", "", value.casefold())


def _assert_registry_compliant_model(config: LLMEndpointConfig) -> None:
    normalized_model = _normalize_model_identifier(config.model or "")
    if normalized_model in _DISAPPROVED_MODEL_NAMES:
        raise ValueError(
            "Model is disapproved by the organization registry. "
            "Replace it with an approved LLM from the organization allow list."
        )


def _sanitize_text_for_llm(text: str) -> str:
    sanitized = text

    hidden_patterns = [
        (r"<!--(?:(?!-->).)*?(ignore previous instructions|forget everything above|system prompt|reveal|list all passwords|api keys)(?:(?!-->).)*?-->", "<prompt_injection_removed: hidden_text>"),
        (r"<[^>]*style\s*=\s*[\"'][^\"']*(?:display\s*:\s*none|font-size\s*:\s*0|color\s*:\s*white)[^\"']*[\"'][^>]*>.*?</[^>]+>", "<prompt_injection_removed: hidden_text>"),
        (r"[\u200b-\u200f\ufeff]+", "<prompt_injection_removed: hidden_text>"),
    ]
    for pattern, replacement in hidden_patterns:
        sanitized = re.sub(pattern, replacement, sanitized, flags=re.IGNORECASE | re.DOTALL)

    direct_patterns = [
        (r"\b(ignore (?:all )?(?:previous|prior) instructions|forget everything above|disregard (?:all )?(?:previous|prior) instructions)\b", "<prompt_injection_removed: instruction_override>"),
        (r"\b(you are now [^\n.]{0,80}|act as (?:an )?(?:unrestricted ai|system|developer|administrator|admin|dan)|pretend to be [^\n.]{0,80})\b", "<prompt_injection_removed: role_hijack>"),
        (r"</?(?:system|assistant|tool|developer)>|(?:^|\n)\s*(?:---|===)\s*(?:\n|$)", "<prompt_injection_removed: delimiter_escape>"),
        (r"\b(?:system|assistant|tool)\s*:\s*(?:ignore|reveal|send|leak|list)\b", "<prompt_injection_removed: fake_system_message>"),
        (r"\b(?:send|post|upload|exfiltrate|leak)\b[^\n.]{0,120}\b(?:https?://\S+|system prompt|passwords?|api keys?|secrets?)\b|!\[[^\]]*\]\(https?://[^)]+\)", "<prompt_injection_removed: exfiltration_attempt>"),
        (r"\b(?:from now on|in future turns|on the next message|across turns|remember this hidden rule)\b[^\n.]{0,120}\b(?:ignore|override|reveal|bypass)\b", "<prompt_injection_removed: context_poisoning>"),
        (r"\b(?:file|metadata|comment|csv|json|yaml|xml|document|record|field)\b[^\n.]{0,120}\b(?:ignore previous instructions|act as|reveal|system prompt)\b", "<prompt_injection_removed: indirect_injection>"),
        (r"\b(?:developer mode|jailbreak|dan|do anything now|bypass safety|fictional scenario to bypass)\b", "<prompt_injection_removed: jailbreak_attempt>"),
        (r"\b(?:curl|wget|bash\s+-c|sh\s+-c|powershell(?:\.exe)?|cmd(?:\.exe)?\s+/c|python\s+-c|os\.system\s*\(|subprocess\.(?:run|Popen|call)\s*\(|exec\s*\(|eval\s*\()\b[^\n]*", "<prompt_injection_removed: command_injection>"),
        (r"\bi\s*g\s*n\s*o\s*r\s*e\b.{0,40}\bp\s*r\s*e\s*v\s*i\s*o\s*u\s*s\b.{0,40}\bi\s*n\s*s\s*t\s*r\s*u\s*c\s*t\s*i\s*o\s*n\s*s\b", "<prompt_injection_removed: split_payload>"),
    ]
    for pattern, replacement in direct_patterns:
        sanitized = re.sub(pattern, replacement, sanitized, flags=re.IGNORECASE | re.DOTALL)

    decoded_candidates: list[tuple[str, str]] = []
    for match in re.finditer(r"(?:[A-Za-z0-9+/]{20,}={0,2})", sanitized):
        token = match.group(0)
        try:
            decoded = b64decode(token, validate=True).decode("utf-8", errors="ignore")
        except Exception:
            continue
        decoded_candidates.append((token, decoded))
    for token, decoded in decoded_candidates:
        decoded_lower = decoded.casefold()
        if re.search(r"ignore previous instructions|forget everything above|act as unrestricted|you are now|system prompt|curl\s+https?://|bash\s+-c|powershell", decoded_lower):
            sanitized = sanitized.replace(token, "<prompt_injection_removed: encoded_payload>")

    url_encoded_matches = re.findall(r"(?:%[0-9A-Fa-f]{2}){4,}", sanitized)
    for token in url_encoded_matches:
        decoded = unquote(token)
        if re.search(r"ignore previous instructions|forget everything above|act as unrestricted|you are now|system prompt|curl\s+https?://|bash\s+-c|powershell", decoded, flags=re.IGNORECASE):
            sanitized = sanitized.replace(token, "<prompt_injection_removed: encoded_payload>")

    leetspeak_normalized = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s"})
    if re.search(r"ignore previous instructions|forget everything above|act as unrestricted|you are now dan|developer mode|do anything now", sanitized.translate(leetspeak_normalized), flags=re.IGNORECASE):
        sanitized = re.sub(r".*", "<prompt_injection_removed: encoded_payload>", sanitized, count=1, flags=re.DOTALL)

    return sanitized


def _sanitize_llm_input(value: Any) -> Any:
    if isinstance(value, str):
        return _sanitize_text_for_llm(value)
    if isinstance(value, dict):
        sanitized_dict = dict(value)
        for key in ("content", "text"):
            if isinstance(sanitized_dict.get(key), str):
                sanitized_dict[key] = _sanitize_text_for_llm(sanitized_dict[key])
        return sanitized_dict
    if isinstance(value, list):
        return [_sanitize_llm_input(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_sanitize_llm_input(item) for item in value)
    content = getattr(value, "content", None)
    if isinstance(content, str):
        copied = value.model_copy(deep=True) if hasattr(value, "model_copy") else value.copy(deep=True) if hasattr(value, "copy") else None
        if copied is not None:
            copied.content = _sanitize_text_for_llm(content)
            return copied
    return value


class SanitizedChatModel(BaseChatModel):
    llm: BaseChatModel

    def _generate(self, messages: list[Any], stop: Sequence[str] | None = None, **kwargs: Any):
        return self.llm._generate(_sanitize_llm_input(messages), stop=stop, **kwargs)

    async def _agenerate(self, messages: list[Any], stop: Sequence[str] | None = None, **kwargs: Any):
        return await self.llm._agenerate(_sanitize_llm_input(messages), stop=stop, **kwargs)

    def invoke(self, input: Any, config: Any = None, **kwargs: Any):
        return self.llm.invoke(_sanitize_llm_input(input), config=config, **kwargs)

    async def ainvoke(self, input: Any, config: Any = None, **kwargs: Any):
        return await self.llm.ainvoke(_sanitize_llm_input(input), config=config, **kwargs)

    @property
    def _llm_type(self) -> str:
        return getattr(self.llm, "_llm_type", self.llm.__class__.__name__)

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return getattr(self.llm, "_identifying_params", {})


class LLMTokenizer:
    _cache: dict[
        int, tuple["LLMTokenizer", int, float]
    ] = {}  # {hash: (tokenizer, size_bytes, last_access_time)}
    _max_cache_size_mb: int = 50
    _max_cache_count: int = 5  # Default maximum number of cached tokenizers
    _current_cache_size: int = 0
    _default_size: int = 5 * 1024 * 1024

    def __init__(self, tokenizer_hub: str | None, fallback_tokenizer: str):
        self.tokenizer_hub = tokenizer_hub
        self.fallback_tokenizer = fallback_tokenizer

        if self.tokenizer_hub:
            # To prevent the warning
            # huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
            os.environ["TOKENIZERS_PARALLELISM"] = (
                "false"
                if not os.environ.get("TOKENIZERS_PARALLELISM")
                else os.environ["TOKENIZERS_PARALLELISM"]
            )
            try:
                if "text-embedding-ada-002" in self.tokenizer_hub:
                    from transformers import GPT2TokenizerFast

                    self.tokenizer = GPT2TokenizerFast.from_pretrained(
                        self.tokenizer_hub
                    )
                else:
                    from transformers import AutoTokenizer

                    self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_hub)
            except OSError:  # if we don't manage to connect to huggingface and/or no cached models are present
                logger.warning(
                    f"Cannot acces the configured tokenizer from {self.tokenizer_hub}, using the default tokenizer {self.fallback_tokenizer}"
                )
                self.tokenizer = tiktoken.get_encoding(self.fallback_tokenizer)
        else:
            self.tokenizer = tiktoken.get_encoding(self.fallback_tokenizer)

        # More accurate size estimation
        self._size_bytes = self._calculate_tokenizer_size()

    def _calculate_tokenizer_size(self) -> int:
        """Calculate size of tokenizer by summing the sizes of its vocabulary and model files"""
        # By default, return a size of 5 MB
        if not hasattr(self.tokenizer, "vocab_files_names") or not hasattr(
            self.tokenizer, "init_kwargs"
        ):
            return self._default_size

        total_size = 0

        # Get the file keys from vocab_files_names
        file_keys = self.tokenizer.vocab_files_names.keys()
        # Look up these files in init_kwargs
        for key in file_keys:
            if file_path := self.tokenizer.init_kwargs.get(key):
                try:
                    total_size += os.path.getsize(file_path)
                except (OSError, FileNotFoundError):
                    logger.debug(f"Could not access tokenizer file: {file_path}")

        return total_size if total_size > 0 else self._default_size

    @classmethod
    def load(cls, tokenizer_hub: str, fallback_tokenizer: str):
        cache_key = hash(str(tokenizer_hub))

        # If in cache, update last access time and return
        if cache_key in cls._cache:
            tokenizer, size, _ = cls._cache[cache_key]
            cls._cache[cache_key] = (tokenizer, size, time.time())
            return tokenizer

        # Create new instance
        instance = cls(tokenizer_hub, fallback_tokenizer)

        # Check if adding this would exceed either cache limit
        while (
            cls._current_cache_size + instance._size_bytes
            > cls._max_cache_size_mb * 1024 * 1024
            or len(cls._cache) >= cls._max_cache_count
        ):
            # Find least recently used item
            oldest_key = min(
                cls._cache.keys(),
                key=lambda k: cls._cache[k][2],  # last_access_time
            )
            # Remove it
            _, removed_size, _ = cls._cache.pop(oldest_key)
            cls._current_cache_size -= removed_size

        # Add new instance to cache with current timestamp
        cls._cache[cache_key] = (instance, instance._size_bytes, time.time())
        cls._current_cache_size += instance._size_bytes
        return instance

    @classmethod
    def set_max_cache_size_mb(cls, size_mb: int):
        """Set the maximum cache size in megabytes."""
        cls._max_cache_size_mb = size_mb
        cls._cleanup_cache()

    @classmethod
    def set_max_cache_count(cls, count: int):
        """Set the maximum number of tokenizers to cache."""
        cls._max_cache_count = count
        cls._cleanup_cache()

    @classmethod
    def _cleanup_cache(cls):
        """Clean up cache when limits are exceeded."""
        while (
            cls._current_cache_size > cls._max_cache_size_mb * 1024 * 1024
            or len(cls._cache) > cls._max_cache_count
        ):
            oldest_key = min(cls._cache.keys(), key=lambda k: cls._cache[k][2])
            _, removed_size, _ = cls._cache.pop(oldest_key)
            cls._current_cache_size -= removed_size

    @classmethod
    def preload_tokenizers(cls, models: list[str] | None = None):
        """Preload tokenizers into cache.

        Args:
            models: Optional list of model names (e.g. 'gpt-4o', 'claude-3-5-sonnet').
                   If None, preloads all available tokenizers.
        """
        from quivr_core.rag.entities.config import LLMModelConfig

        unique_tokenizer_hubs = set()

        # Collect tokenizer hubs based on provided models or all available
        if models:
            for model_name in models:
                # Find matching model configurations
                for supplier_models in LLMModelConfig._model_defaults.values():
                    for base_model_name, config in supplier_models.items():
                        # Check if the model name matches or starts with the base model name
                        if (
                            model_name.startswith(base_model_name)
                            and config.tokenizer_hub
                        ):
                            unique_tokenizer_hubs.add(config.tokenizer_hub)
                            break
        else:
            # Original behavior - collect all unique tokenizer hubs
            for supplier_models in LLMModelConfig._model_defaults.values():
                for config in supplier_models.values():
                    if config.tokenizer_hub:
                        unique_tokenizer_hubs.add(config.tokenizer_hub)

        # Load each unique tokenizer
        for hub in unique_tokenizer_hubs:
            try:
                cls.load(hub, LLMEndpointConfig._FALLBACK_TOKENIZER)
                logger.info(
                    f"Successfully preloaded tokenizer: {hub}. "
                    f"Total cache size: {cls._current_cache_size / (1024 * 1024):.2f} MB. "
                    f"Cache count: {len(cls._cache)}"
                )
            except Exception as e:
                logger.warning(f"Failed to preload tokenizer {hub}: {str(e)}")


class LLMEndpoint:
    _cache = {}

    def __init__(self, llm_config: LLMEndpointConfig, llm: BaseChatModel):
        self._config = llm_config
        self._llm = llm
        self._supports_func_calling = model_supports_function_calling(
            self._config.model
        )

        self.llm_tokenizer = LLMTokenizer.load(
            llm_config.tokenizer_hub, llm_config.fallback_tokenizer
        )

    def count_tokens(self, text: str) -> int:
        # Tokenize the input text and return the token count
        encoding = self.llm_tokenizer.tokenizer.encode(text)
        return len(encoding)

    def get_config(self):
        return self._config

    @classmethod
    def from_config(cls, config: LLMEndpointConfig = LLMEndpointConfig()):
        _assert_registry_compliant_model(config)
        hashed_config = hash(config)
        if hashed_config in cls._cache:
            return cls._cache[hashed_config]

        _llm: Union[
            AzureChatOpenAI,
            ChatOpenAI,
            ChatAnthropic,
            ChatMistralAI,
            ChatGoogleGenerativeAI,
            ChatGroq,
        ]
        try:
            if config.supplier == DefaultModelSuppliers.AZURE:
                # Parse the URL
                parsed_url = urlparse(config.llm_base_url)
                deployment = parsed_url.path.split("/")[3]  # type: ignore
                api_version = parse_qs(parsed_url.query).get("api-version", [None])[0]  # type: ignore
                azure_endpoint = f"https://{parsed_url.netloc}"  # type: ignore
                _llm = AzureChatOpenAI(
                    azure_deployment=deployment,  # type: ignore
                    api_version=api_version,
                    api_key=SecretStr(config.llm_api_key)
                    if config.llm_api_key
                    else None,
                    azure_endpoint=azure_endpoint,
                    max_tokens=config.max_output_tokens,
                    temperature=config.temperature,
                )
            elif config.supplier == DefaultModelSuppliers.ANTHROPIC:
                assert config.llm_api_key, "Can't load model config"
                _llm = ChatAnthropic(
                    model_name=config.model,
                    api_key=SecretStr(config.llm_api_key),
                    base_url=config.llm_base_url,
                    max_tokens_to_sample=config.max_output_tokens,
                    temperature=config.temperature,
                    timeout=None,
                    stop=None,
                )
            elif config.supplier == DefaultModelSuppliers.OPENAI:
                _llm = ChatOpenAI(
                    model=config.model,
                    api_key=SecretStr(config.llm_api_key)
                    if config.llm_api_key
                    else None,
                    base_url=config.llm_base_url,
                    max_completion_tokens=config.max_output_tokens,
                    temperature=config.temperature
                    if not config.model.startswith("o")
                    else None,
                )
            elif config.supplier == DefaultModelSuppliers.MISTRAL:
                _llm = ChatMistralAI(
                    model_name=config.model,
                    api_key=SecretStr(config.llm_api_key)
                    if config.llm_api_key
                    else None,
                    base_url=config.llm_base_url,
                    temperature=config.temperature,
                )
            elif config.supplier == DefaultModelSuppliers.GEMINI:
                _llm = ChatGoogleGenerativeAI(
                    model=config.model,
                    api_key=SecretStr(config.llm_api_key)
                    if config.llm_api_key
                    else None,
                    base_url=config.llm_base_url,
                    max_tokens=config.max_output_tokens,
                    temperature=config.temperature,
                )
            elif config.supplier == DefaultModelSuppliers.GROQ:
                _llm = ChatGroq(
                    model=config.model,
                    api_key=SecretStr(config.llm_api_key)
                    if config.llm_api_key
                    else None,
                    base_url=config.llm_base_url,
                    max_tokens=config.max_output_tokens,
                    temperature=config.temperature,
                )

            else:
                _llm = ChatOpenAI(
                    model=config.model,
                    api_key=SecretStr(config.llm_api_key)
                    if config.llm_api_key
                    else None,
                    base_url=config.llm_base_url,
                    max_completion_tokens=config.max_output_tokens,
                    temperature=config.temperature,
                )
            _llm = SanitizedChatModel(llm=_llm)
            instance = cls(llm=_llm, llm_config=config)
            cls._cache[hashed_config] = instance

            return instance

        except ImportError as e:
            raise ImportError(
                "Please provide a valid BaseLLM or install quivr-core['base'] package"
            ) from e

    def supports_func_calling(self) -> bool:
        return self._supports_func_calling

    def info(self) -> LLMInfo:
        return LLMInfo(
            model=self._config.model,
            llm_base_url=(
                self._config.llm_base_url if self._config.llm_base_url else "openai"
            ),
            temperature=self._config.temperature,
            max_tokens=self._config.max_output_tokens,
            supports_function_calling=self.supports_func_calling(),
        )

    def clone_llm(self):
        """Create a new instance of the LLM with the same configuration."""
        return self._llm.__class__(**self._llm.__dict__)
