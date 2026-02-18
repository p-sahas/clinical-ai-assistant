"""
Configuration management for Agentic RAG Systems.

Handles YAML config loading, environment variable validation,
token counting, and configuration utilities.
"""

import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv
import tiktoken

# Load environment variables
load_dotenv()


def _find_file(name: str) -> Optional[Path]:
    """Find a file in cwd, parent, or up the tree."""
    for base in [Path.cwd(), Path.cwd().parent]:
        p = base / name
        if p.exists():
            return p.resolve()
    current = Path.cwd()
    for _ in range(5):
        p = current / name
        if p.exists():
            return p.resolve()
        current = current.parent
        if current == current.parent:
            break
    return None


class ConfigManager:
    # Manages system configuration and environment validation.

    def __init__(self, config_path: str = "config"):
        """
        Initialize configuration manager.

        Args:
            config_path: "config" (load config/params.yaml + config/models.yaml)
                        or path to single YAML file (e.g. "config.yaml")
        """
        self._config_dir: Optional[Path] = None
        self.config_path: Optional[Path] = None
        self.config: Dict[str, Any] = {}
        self._models: Dict[str, Any] = {}
        if config_path.strip("/") == "config":
            self._load_config_dir()
        else:
            self.config_path = self._resolve_config_path(config_path)
            self._load_single_file()
        self._validate_env()

    def _resolve_config_path(self, config_path: str) -> Path:
        """Resolve config file path (cwd, parent, or up the tree)."""
        path = Path(config_path)
        if path.is_absolute() and path.exists():
            return path
        found = _find_file(config_path)
        if found:
            return found
        return path

    def _load_config_dir(self) -> None:
        """Load config/params.yaml and config/models.yaml; resolve model from provider + tier."""
        params_file = _find_file("config/params.yaml")
        if not params_file:
            # Fallback to root config.yaml
            single = _find_file("config.yaml")
            if single:
                self.config_path = single
                self._load_single_file()
                return
            raise FileNotFoundError(
                "Config not found: config/params.yaml or config.yaml\n"
                f"Searched from: {Path.cwd()}"
            )
        self._config_dir = params_file.parent
        with open(params_file, "r") as f:
            self.config = yaml.safe_load(f) or {}
        models_file = self._config_dir / "models.yaml"
        if models_file.exists():
            with open(models_file, "r") as f:
                self._models = yaml.safe_load(f) or {}
        provider = (self.config.get("provider") or {}
                    ).get("default", "openrouter")
        tier = (self.config.get("provider") or {}).get("tier", "general")
        embed_tier = (self.config.get("embedding")
                      or {}).get("tier", "default")
        # Resolve chat model
        chat_model = (
            (self._models.get(provider) or {}).get("chat") or {}
        ).get(tier)
        if chat_model:
            self.config["model"] = chat_model
        # Resolve embedding model from models.yaml for openai and openrouter
        if provider in ["openai", "openrouter"]:
            embed_model = (
                (self._models.get(provider) or {}).get("embedding") or {}
            ).get(embed_tier)
            if embed_model and "rag" in self.config:
                self.config["rag"]["embed_model"] = embed_model
        # Flatten llm into top-level for backward compat
        llm = self.config.get("llm") or {}
        if "temperature" not in self.config and "temperature" in llm:
            self.config["temperature"] = llm["temperature"]
        if "max_tokens" not in self.config and "max_tokens" in llm:
            self.config["max_tokens"] = llm["max_tokens"]
        # RAG: use chunking + retrieval from params
        chunking = self.config.get("chunking") or {}
        if chunking and "rag" in self.config:
            self.config["rag"]["chunk_size"] = chunking.get("chunk_size", 800)
            self.config["rag"]["overlap"] = chunking.get("chunk_overlap", 100)
        retrieval = self.config.get("retrieval") or {}
        if retrieval and "rag" in self.config:
            self.config["rag"]["max_k"] = retrieval.get("top_k", 4)
            # Keep rag.score_threshold if already set (e.g. 0.18); else use retrieval.similarity_threshold
            if "score_threshold" not in self.config["rag"]:
                self.config["rag"]["score_threshold"] = retrieval.get(
                    "similarity_threshold", 0.5
                )
        # data paths
        paths = self.config.get("paths") or {}
        if paths and "data" not in self.config:
            self.config["data"] = {
                "docs_directory": paths.get("docs_directory", "./data/medical_guides"),
                "bookings_file": paths.get("bookings_file", "./data/bookings.json"),
                "cache_directory": paths.get("cache_dir", "./store"),
            }
        if "rag" in self.config and paths:
            self.config["rag"]["cache_path"] = paths.get(
                "cache_dir", "./store")

    def _load_single_file(self) -> None:
        """Load single YAML config file (e.g. config.yaml)."""
        if not self.config_path or not self.config_path.exists():
            raise FileNotFoundError(
                f"Config file not found: {self.config_path}\n"
                f"Searched from: {Path.cwd()}"
            )
        with open(self.config_path, "r") as f:
            self.config = yaml.safe_load(f) or {}

    def _load_config(self) -> None:
        """Legacy: dispatch to dir or single file (used only if get_config passed path)."""
        if self._config_dir is not None:
            return
        if self.config_path and self.config_path.exists():
            self._load_single_file()

    def _validate_env(self) -> None:
        """Validate required environment variables based on provider."""
        missing = []
        provider = (self.config.get("provider") or {}
                    ).get("default", "openrouter")
        if provider == "openrouter":
            if not os.getenv("OPENROUTER_API_KEY"):
                missing.append("OPENROUTER_API_KEY")
        else:
            if not os.getenv("OPENAI_API_KEY"):
                missing.append("OPENAI_API_KEY")
        if not os.getenv("TAVILY_API_KEY"):
            missing.append("TAVILY_API_KEY")
        if missing:
            raise EnvironmentError(
                f"Missing required environment variables: {', '.join(missing)}\n"
                "Set them in .env (e.g. OPENROUTER_API_KEY or OPENAI_API_KEY, TAVILY_API_KEY)."
            )

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.

        Args:
            key_path: Dot-separated path (e.g., 'rag.chunk_size')
            default: Default value if key not found

        Returns:
            Configuration value
        """
        keys = key_path.split('.')
        value = self.config

        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
                if value is None:
                    return default
            else:
                return default

        return value

    def dump(self) -> Dict[str, Any]:
        """
        Return complete configuration as dictionary.

        Returns:
            Full configuration dictionary
        """
        return self.config.copy()


class TokenCounter:
    """Utility for counting tokens in text."""

    def __init__(self, model: str = "gpt-4o-mini"):
        """
        Initialize token counter.

        Args:
            model: Model name for tokenizer selection
        """
        try:
            # Get appropriate encoding for model
            if "gpt-4" in model:
                self.encoding = tiktoken.encoding_for_model("gpt-4")
            elif "gpt-3.5" in model:
                self.encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
            else:
                self.encoding = tiktoken.get_encoding("cl100k_base")
        except Exception:
            # Fallback to cl100k_base encoding
            self.encoding = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.

        Args:
            text: Input text

        Returns:
            Token count
        """
        return len(self.encoding.encode(text))

    def count_messages_tokens(self, messages: list) -> int:
        """
        Count tokens in message list (for chat format).

        Args:
            messages: List of message dictionaries

        Returns:
            Total token count
        """
        token_count = 0
        for message in messages:
            # Add tokens for role and content
            token_count += 4  # Format overhead per message
            token_count += self.count_tokens(message.get("role", ""))
            token_count += self.count_tokens(message.get("content", ""))

        token_count += 2  # Overhead for reply priming
        return token_count


def get_api_key(service: str) -> str:
    """
    Get API key for specified service.

    Args:
        service: Service name (openai, openrouter, tavily)

    Returns:
        API key string

    Raises:
        ValueError: If API key not found
    """
    key_map = {
        "openai": "OPENAI_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
        "tavily": "TAVILY_API_KEY",
    }
    env_var = key_map.get(service.lower())
    if not env_var:
        raise ValueError(f"Unknown service: {service}")
    api_key = os.getenv(env_var)
    if not api_key:
        raise ValueError(
            f"API key not found for {service}: set {env_var} in .env")
    return api_key


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration dictionary.

    Args:
        config: Configuration dictionary to validate

    Returns:
        True if valid

    Raises:
        ValueError: If configuration is invalid
    """
    required_keys = ["rag", "web", "output"]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")
    rag = config["rag"]
    for key in ["chunk_size", "max_k"]:
        if key not in rag:
            raise ValueError(f"Missing required RAG config key: rag.{key}")
    web_required = ["provider", "max_results"]
    for key in web_required:
        if key not in config["web"]:
            raise ValueError(f"Missing required web config key: {key}")
    return True


# Global configuration instance
_config_instance: Optional[ConfigManager] = None


def get_config(config_path: str = "config") -> ConfigManager:
    """
    Get global configuration instance (singleton pattern).

    Args:
        config_path: Path to configuration file

    Returns:
        ConfigManager instance
    """
    global _config_instance

    if _config_instance is None:
        _config_instance = ConfigManager(config_path)

    return _config_instance
