"""
LLM service providers and tool registry for Agentic RAG Systems.

Provides OpenAI integration and maintains tool registry for agent interactions.
Only web_search is registered as an external tool (RAG is internal).
"""

import time
from typing import Dict, Any, List, Optional, Callable
from openai import OpenAI
from utils.config import get_config, get_api_key, TokenCounter


class EmbeddingProvider:
    """Base class for embedding providers."""

    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors
        """
        raise NotImplementedError


class OpenRouterEmbeddingProvider(EmbeddingProvider):
    """OpenRouter embedding provider."""

    def __init__(self, model: str = "openai/text-embedding-3-large"):
        self.api_key = get_api_key("openrouter")
        self.client = OpenAI(api_key=self.api_key,
                             base_url="https://openrouter.ai/api/v1")
        self.model = model

    def embed(self, texts: List[str]) -> List[List[float]]:
        safe_texts = [t.strip() if t else " " for t in texts]
        response = self.client.embeddings.create(
            model=self.model, input=safe_texts)
        return [item.embedding for item in response.data]


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI embedding provider."""

    def __init__(self, model: str = "text-embedding-3-small"):
        self.api_key = get_api_key("openai")
        self.client = OpenAI(api_key=self.api_key)
        self.model = model

    def embed(self, texts: List[str]) -> List[List[float]]:
        safe_texts = [t.strip() if t else " " for t in texts]
        response = self.client.embeddings.create(
            model=self.model, input=safe_texts)
        return [item.embedding for item in response.data]


class SentenceTransformerEmbeddingProvider(EmbeddingProvider):
    """Local SentenceTransformer embedding provider."""

    def __init__(self, model: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model)

    def embed(self, texts: List[str]) -> List[List[float]]:
        safe_texts = [t.strip() if t else " " for t in texts]
        embeddings = self.model.encode(safe_texts)
        return [emb.tolist() for emb in embeddings]


class LLMProvider:
    """Base class for LLM providers."""

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate response from LLM.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            **kwargs: Additional generation parameters

        Returns:
            Dictionary with response, tokens, and timing info
        """
        raise NotImplementedError


class OpenRouterProvider(LLMProvider):
    """OpenRouter API provider – unified access to multiple LLM backends."""

    def __init__(
        self,
        model: str = "openai/gpt-4o-mini",
        temperature: float = 0.0,
        max_tokens: int = 2000,
        base_url: str = "https://openrouter.ai/api/v1",
    ):
        self.api_key = get_api_key("openrouter")
        self.client = OpenAI(api_key=self.api_key, base_url=base_url)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.token_counter = TokenCounter(
            "gpt-4o-mini")  # fallback for token counting

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        start_time = time.time()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature if temperature is not None else self.temperature,
            max_tokens=max_tokens or self.max_tokens,
            **kwargs
        )
        latency_ms = int((time.time() - start_time) * 1000)
        msg = response.choices[0].message
        return {
            "response": msg.content,
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
            "latency_ms": latency_ms,
            "model": self.model,
        }

    def generate_with_tools(
        self,
        prompt: str,
        tools: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        start_time = time.time()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            **kwargs
        )
        latency_ms = int((time.time() - start_time) * 1000)
        message = response.choices[0].message
        return {
            "response": message.content,
            "tool_calls": getattr(message, "tool_calls", None),
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
            "latency_ms": latency_ms,
            "model": self.model,
        }


class OpenAIProvider(LLMProvider):
    """OpenAI API provider for GPT models."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.2,
        max_tokens: int = 512
    ):
        """
        Initialize OpenAI provider.

        Args:
            model: Model name
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
        """
        self.api_key = get_api_key("openai")
        self.client = OpenAI(api_key=self.api_key)
        self.model = model.replace("openai_compatible:", "")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.token_counter = TokenCounter(self.model)

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate response using OpenAI API.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Override default temperature
            max_tokens: Override default max tokens
            **kwargs: Additional parameters

        Returns:
            Dictionary with:
                - response: Generated text
                - prompt_tokens: Input token count
                - completion_tokens: Output token count
                - total_tokens: Total token count
                - latency_ms: Generation time in milliseconds
        """
        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Track timing
        start_time = time.time()

        # Call API
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature or self.temperature,
            max_tokens=max_tokens or self.max_tokens,
            **kwargs
        )

        end_time = time.time()
        latency_ms = int((end_time - start_time) * 1000)

        # Extract response and metrics
        result = {
            "response": response.choices[0].message.content,
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
            "latency_ms": latency_ms,
            "model": self.model
        }

        return result

    def generate_with_tools(
        self,
        prompt: str,
        tools: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate response with tool calling support.

        Args:
            prompt: User prompt
            tools: List of tool definitions in OpenAI format
            system_prompt: Optional system prompt
            **kwargs: Additional parameters

        Returns:
            Dictionary with response and tool calls
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        start_time = time.time()

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            **kwargs
        )

        end_time = time.time()
        latency_ms = int((end_time - start_time) * 1000)

        message = response.choices[0].message

        result = {
            "response": message.content,
            "tool_calls": message.tool_calls if hasattr(message, 'tool_calls') else None,
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
            "latency_ms": latency_ms,
            "model": self.model
        }

        return result


class DummyLocalProvider(LLMProvider):
    """Dummy local provider for testing (CPU-friendly)."""

    def __init__(self):
        """Initialize dummy provider."""
        self.model = "dummy_local"

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate dummy response.

        Returns:
            Dummy response dictionary
        """
        response_text = (
            "This is a dummy response for testing. "
            "In production, this would be a real LLM response based on the prompt."
        )

        return {
            "response": response_text,
            "prompt_tokens": 50,
            "completion_tokens": 20,
            "total_tokens": 70,
            "latency_ms": 100,
            "model": "dummy_local"
        }


class ToolRegistry:
    """
    Registry for external agent tools.

    Note: RAG is NOT registered here - it's internal memory/cognition.
    Only web_search is registered as an external action tool.
    """

    def __init__(self):
        """Initialize tool registry."""
        self._tools: Dict[str, Callable] = {}
        self._tool_definitions: List[Dict[str, Any]] = []

    def register_tool(
        self,
        name: str,
        func: Callable,
        description: str,
        parameters: Dict[str, Any]
    ) -> None:
        """
        Register a tool for agent use.

        Args:
            name: Tool name
            func: Tool function
            description: Tool description for LLM
            parameters: JSON schema for tool parameters
        """
        self._tools[name] = func

        # Create OpenAI function definition
        tool_def = {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters
            }
        }

        self._tool_definitions.append(tool_def)

    def get_tool(self, name: str) -> Optional[Callable]:
        """
        Get tool function by name.

        Args:
            name: Tool name

        Returns:
            Tool function or None
        """
        return self._tools.get(name)

    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """
        Get tool definitions for LLM (OpenAI format).

        Returns:
            List of tool definitions
        """
        return self._tool_definitions.copy()

    def list_tools(self) -> List[str]:
        """
        List registered tool names.

        Returns:
            List of tool names
        """
        return list(self._tools.keys())

    def get_tools_header(self) -> str:
        """
        Get text description of available tools for prompts.

        Returns:
            Formatted tool header string
        """
        if not self._tools:
            return "No external tools available."

        header = "Available External Tools:\n\n"
        for tool_def in self._tool_definitions:
            func_def = tool_def["function"]
            header += f"- {func_def['name']}: {func_def['description']}\n"

        header += "\nNote: Internal RAG retrieval is always available but not listed as a tool."

        return header


def create_llm_provider(config: Optional[Dict[str, Any]] = None) -> LLMProvider:
    """
    Factory function to create LLM provider based on configuration.

    Uses config/params.yaml + config/models.yaml when available (provider.default + tier).
    Falls back to config.yaml or explicit config dict.
    """
    if config is None:
        cfg = get_config()
        model = cfg.get("model", "gpt-4o-mini")
        temperature = cfg.get("temperature", 0.2)
        max_tokens = cfg.get("max_tokens", 512)
        provider = cfg.get("provider.default") or (
            cfg.get("provider") or {}).get("default") or "openrouter"
        openrouter_base = (cfg.get("provider") or {}).get(
            "openrouter_base_url", "https://openrouter.ai/api/v1")
    else:
        model = config.get("model", "gpt-4o-mini")
        temperature = config.get("temperature", 0.2)
        max_tokens = config.get("max_tokens", 512)
        provider = ((config.get("provider") or {}).get("default") or "openrouter") if isinstance(
            config.get("provider"), dict) else "openrouter"
        openrouter_base = (config.get("provider") or {}).get(
            "openrouter_base_url", "https://openrouter.ai/api/v1")

    if "dummy" in (model or "").lower():
        return DummyLocalProvider()
    if provider == "openrouter":
        return OpenRouterProvider(
            model=model or "openai/gpt-4o-mini",
            temperature=temperature,
            max_tokens=max_tokens,
            base_url=openrouter_base,
        )
    if provider != "openai" and provider != "openrouter":
        raise ValueError(
            f"Unknown provider: {provider}. Use openrouter or openai.")
    # OpenAI when explicitly set to openai
    return OpenAIProvider(
        model=(model or "gpt-4o-mini").replace("openai_compatible:", ""),
        temperature=temperature,
        max_tokens=max_tokens,
    )


def create_embedding_provider(config: Optional[Dict[str, Any]] = None) -> EmbeddingProvider:
    """
    Factory function to create embedding provider based on configuration.

    Uses config/params.yaml + config/models.yaml when available.
    Falls back to sentence-transformers if no API keys.
    """
    if config is None:
        cfg = get_config()
        embed_model = cfg.get("rag.embed_model", "all-MiniLM-L6-v2")
        provider = cfg.get("provider.default") or (
            cfg.get("provider") or {}).get("default") or "openrouter"
    else:
        embed_model = config.get("rag.embed_model", "all-MiniLM-L6-v2")
        provider = ((config.get("provider") or {}).get("default") or "openrouter") if isinstance(
            config.get("provider"), dict) else "openrouter"

    # If embed_model is a full model name like openai/text-embedding-3-large, use API
    if embed_model.startswith("openai/"):
        if provider == "openrouter":
            return OpenRouterEmbeddingProvider(model=embed_model)
        elif provider == "openai":
            return OpenAIEmbeddingProvider(model=embed_model.replace("openai/", ""))
        else:
            # Fallback to local
            return SentenceTransformerEmbeddingProvider()
    else:
        # Assume local model
        return SentenceTransformerEmbeddingProvider(model=embed_model)


# Global tool registry instance
_tool_registry: Optional[ToolRegistry] = None


def get_tool_registry() -> ToolRegistry:
    """
    Get global tool registry instance (singleton pattern).

    Returns:
        ToolRegistry instance
    """
    global _tool_registry

    if _tool_registry is None:
        _tool_registry = ToolRegistry()

    return _tool_registry
