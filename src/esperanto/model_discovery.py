"""Static model discovery functions for all providers.

This module provides static discovery functions that can be used to list available
models from providers without instantiating provider classes.
"""

import hashlib
import os
from typing import Callable, Dict, List, Optional

import httpx

from esperanto.common_types import Model
from esperanto.utils import ModelCache

# Global cache instance for model discovery
_model_cache = ModelCache()


def _create_cache_key(provider: str, **config) -> str:
    """Create a unique cache key for a provider configuration.

    Args:
        provider: Provider name
        **config: Configuration parameters (api_key, base_url, etc.)

    Returns:
        Unique cache key string
    """
    # Hash the API key if present (for privacy)
    config_copy = config.copy()
    if "api_key" in config_copy and config_copy["api_key"]:
        api_key_hash = hashlib.sha256(config_copy["api_key"].encode()).hexdigest()[:16]
        config_copy["api_key"] = api_key_hash

    # Create deterministic string from config
    config_str = "&".join(f"{k}={v}" for k, v in sorted(config_copy.items()) if v is not None)
    return f"{provider}:{config_str}"


def get_openai_models(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    organization: Optional[str] = None,
    model_type: Optional[str] = "language",
) -> List[Model]:
    """Get available models from OpenAI.

    Args:
        api_key: OpenAI API key (or OPENAI_API_KEY env var)
        base_url: Base URL for API (default: https://api.openai.com/v1)
        organization: OpenAI organization ID
        model_type: Type of models to return: 'language', 'embedding', 'speech_to_text',
                   'text_to_speech', or None for all models

    Returns:
        List of available models

    Raises:
        ValueError: If API key is not provided
        RuntimeError: If API request fails
    """
    # Get API key
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key not found. Provide api_key or set OPENAI_API_KEY environment variable.")

    # Set defaults
    base_url = base_url or "https://api.openai.com/v1"

    # Check cache
    cache_key = _create_cache_key("openai", api_key=api_key, base_url=base_url, model_type=model_type)
    cached_models = _model_cache.get(cache_key)
    if cached_models is not None:
        return cached_models

    # Prepare headers
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if organization:
        headers["OpenAI-Organization"] = organization

    # Make request
    try:
        response = httpx.get(
            f"{base_url}/models",
            headers=headers,
            timeout=60.0
        )

        if response.status_code >= 400:
            try:
                error_data = response.json()
                error_message = error_data.get("error", {}).get("message", f"HTTP {response.status_code}")
            except Exception:
                error_message = f"HTTP {response.status_code}: {response.text}"
            raise RuntimeError(f"OpenAI API error: {error_message}")

        models_data = response.json()

        # Filter models based on type
        all_models = []
        for model in models_data["data"]:
            model_id = model["id"]

            # Determine if this model matches the requested type
            include = False
            if model_type is None:
                include = True
            elif model_type == "language" and model_id.startswith("gpt"):
                include = True
            elif model_type == "embedding" and model_id.startswith("text-embedding"):
                include = True
            elif model_type == "speech_to_text" and model_id.startswith("whisper"):
                include = True
            elif model_type == "text_to_speech" and model_id.startswith("tts"):
                include = True

            if include:
                all_models.append(Model(
                    id=model_id,
                    owned_by=model.get("owned_by", "openai"),
                    context_window=model.get("context_window", None),
                ))

        # Cache results
        _model_cache.set(cache_key, all_models)

        return all_models

    except httpx.HTTPError as e:
        raise RuntimeError(f"Failed to fetch OpenAI models: {e}")


def get_anthropic_models(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> List[Model]:
    """Get available models from Anthropic.

    Note: Anthropic doesn't provide a models API endpoint, so this returns
    a hardcoded list of known models.

    Args:
        api_key: Anthropic API key (not used, kept for consistency)
        base_url: Base URL for API (not used, kept for consistency)

    Returns:
        List of known Anthropic models
    """
    # Check cache
    cache_key = _create_cache_key("anthropic")
    cached_models = _model_cache.get(cache_key)
    if cached_models is not None:
        return cached_models

    # Hardcoded list of known Anthropic models (they don't have a models API)
    models = [
        Model(id="claude-3-5-sonnet-20241022", owned_by="anthropic", context_window=200000),
        Model(id="claude-3-5-sonnet-20240620", owned_by="anthropic", context_window=200000),
        Model(id="claude-3-5-haiku-20241022", owned_by="anthropic", context_window=200000),
        Model(id="claude-3-opus-20240229", owned_by="anthropic", context_window=200000),
        Model(id="claude-3-sonnet-20240229", owned_by="anthropic", context_window=200000),
        Model(id="claude-3-haiku-20240307", owned_by="anthropic", context_window=200000),
    ]

    # Cache results
    _model_cache.set(cache_key, models)

    return models


def get_google_models(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> List[Model]:
    """Get available models from Google Gemini.

    Args:
        api_key: Google API key (or GOOGLE_API_KEY/GEMINI_API_KEY env var)
        base_url: Base URL for API (default: https://generativelanguage.googleapis.com/v1beta)

    Returns:
        List of available models

    Raises:
        ValueError: If API key is not provided
        RuntimeError: If API request fails
    """
    # Get API key
    api_key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Google API key not found. Provide api_key or set GOOGLE_API_KEY environment variable.")

    # Set defaults
    base_url = base_url or "https://generativelanguage.googleapis.com/v1beta"

    # Check cache
    cache_key = _create_cache_key("google", api_key=api_key, base_url=base_url)
    cached_models = _model_cache.get(cache_key)
    if cached_models is not None:
        return cached_models

    # Make request
    try:
        response = httpx.get(
            f"{base_url}/models",
            params={"key": api_key},
            timeout=60.0
        )

        if response.status_code >= 400:
            try:
                error_data = response.json()
                error_message = error_data.get("error", {}).get("message", f"HTTP {response.status_code}")
            except Exception:
                error_message = f"HTTP {response.status_code}: {response.text}"
            raise RuntimeError(f"Google API error: {error_message}")

        models_data = response.json()

        # Parse models - Google returns models with "name" field like "models/gemini-1.5-pro"
        all_models = []
        for model in models_data.get("models", []):
            model_name = model.get("name", "")
            # Extract just the model ID (remove "models/" prefix)
            model_id = model_name.split("/")[-1] if "/" in model_name else model_name

            # Only include generative models (skip embedding models here)
            if model_id and "generateContent" in model.get("supportedGenerationMethods", []):
                all_models.append(Model(
                    id=model_id,
                    owned_by="google",
                    context_window=model.get("inputTokenLimit", None),
                ))

        # Cache results
        _model_cache.set(cache_key, all_models)

        return all_models

    except httpx.HTTPError as e:
        raise RuntimeError(f"Failed to fetch Google models: {e}")


def get_mistral_models(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> List[Model]:
    """Get available models from Mistral AI.

    Args:
        api_key: Mistral API key (or MISTRAL_API_KEY env var)
        base_url: Base URL for API (default: https://api.mistral.ai/v1)

    Returns:
        List of available models

    Raises:
        ValueError: If API key is not provided
        RuntimeError: If API request fails
    """
    # Get API key
    api_key = api_key or os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError("Mistral API key not found. Provide api_key or set MISTRAL_API_KEY environment variable.")

    # Set defaults
    base_url = base_url or "https://api.mistral.ai/v1"

    # Check cache
    cache_key = _create_cache_key("mistral", api_key=api_key, base_url=base_url)
    cached_models = _model_cache.get(cache_key)
    if cached_models is not None:
        return cached_models

    # Prepare headers
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # Make request
    try:
        response = httpx.get(
            f"{base_url}/models",
            headers=headers,
            timeout=60.0
        )

        if response.status_code >= 400:
            try:
                error_data = response.json()
                error_message = error_data.get("message", f"HTTP {response.status_code}")
            except Exception:
                error_message = f"HTTP {response.status_code}: {response.text}"
            raise RuntimeError(f"Mistral API error: {error_message}")

        models_data = response.json()

        # Parse models
        all_models = []
        for model in models_data.get("data", []):
            all_models.append(Model(
                id=model["id"],
                owned_by=model.get("owned_by", "mistral"),
                context_window=model.get("max_context_length", None),
            ))

        # Cache results
        _model_cache.set(cache_key, all_models)

        return all_models

    except httpx.HTTPError as e:
        raise RuntimeError(f"Failed to fetch Mistral models: {e}")


def get_groq_models(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> List[Model]:
    """Get available models from Groq.

    Args:
        api_key: Groq API key (or GROQ_API_KEY env var)
        base_url: Base URL for API (default: https://api.groq.com/openai/v1)

    Returns:
        List of available models

    Raises:
        ValueError: If API key is not provided
        RuntimeError: If API request fails
    """
    # Get API key
    api_key = api_key or os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("Groq API key not found. Provide api_key or set GROQ_API_KEY environment variable.")

    # Set defaults
    base_url = base_url or "https://api.groq.com/openai/v1"

    # Check cache
    cache_key = _create_cache_key("groq", api_key=api_key, base_url=base_url)
    cached_models = _model_cache.get(cache_key)
    if cached_models is not None:
        return cached_models

    # Prepare headers
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # Make request
    try:
        response = httpx.get(
            f"{base_url}/models",
            headers=headers,
            timeout=60.0
        )

        if response.status_code >= 400:
            try:
                error_data = response.json()
                error_message = error_data.get("error", {}).get("message", f"HTTP {response.status_code}")
            except Exception:
                error_message = f"HTTP {response.status_code}: {response.text}"
            raise RuntimeError(f"Groq API error: {error_message}")

        models_data = response.json()

        # Parse models
        all_models = []
        for model in models_data.get("data", []):
            all_models.append(Model(
                id=model["id"],
                owned_by=model.get("owned_by", "groq"),
                context_window=model.get("context_window", None),
            ))

        # Cache results
        _model_cache.set(cache_key, all_models)

        return all_models

    except httpx.HTTPError as e:
        raise RuntimeError(f"Failed to fetch Groq models: {e}")


def get_ollama_models(
    base_url: Optional[str] = None,
) -> List[Model]:
    """Get available models from Ollama.

    Args:
        base_url: Base URL for API (default: http://localhost:11434)

    Returns:
        List of available models

    Raises:
        RuntimeError: If API request fails
    """
    # Set defaults
    base_url = base_url or os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434"

    # Check cache
    cache_key = _create_cache_key("ollama", base_url=base_url)
    cached_models = _model_cache.get(cache_key)
    if cached_models is not None:
        return cached_models

    # Make request
    try:
        response = httpx.get(
            f"{base_url}/api/tags",
            timeout=60.0
        )

        if response.status_code >= 400:
            raise RuntimeError(f"Ollama API error: HTTP {response.status_code}")

        models_data = response.json()

        # Parse models
        all_models = []
        for model in models_data.get("models", []):
            all_models.append(Model(
                id=model["name"],
                owned_by="ollama",
                context_window=None,  # Ollama doesn't provide this in the API
            ))

        # Cache results
        _model_cache.set(cache_key, all_models)

        return all_models

    except httpx.HTTPError as e:
        raise RuntimeError(f"Failed to fetch Ollama models: {e}")


def get_jina_models(
    api_key: Optional[str] = None,
) -> List[Model]:
    """Get available models from Jina AI.

    Note: Jina doesn't provide a models API endpoint, so this returns
    a hardcoded list of known models.

    Args:
        api_key: Jina API key (not used, kept for consistency)

    Returns:
        List of known Jina models
    """
    # Check cache
    cache_key = _create_cache_key("jina")
    cached_models = _model_cache.get(cache_key)
    if cached_models is not None:
        return cached_models

    # Hardcoded list of known Jina models
    models = [
        # Embedding models
        Model(id="jina-embeddings-v4", owned_by="jina", context_window=8192),
        Model(id="jina-embeddings-v3", owned_by="jina", context_window=8192),
        Model(id="jina-embeddings-v2-base-en", owned_by="jina", context_window=8192),
        Model(id="jina-embeddings-v2-base-de", owned_by="jina", context_window=8192),
        Model(id="jina-embeddings-v2-base-zh", owned_by="jina", context_window=8192),
        Model(id="jina-embeddings-v2-base-code", owned_by="jina", context_window=8192),
        # Reranker models
        Model(id="jina-reranker-v2-base-multilingual", owned_by="jina", context_window=1024),
        Model(id="jina-reranker-v1-base-en", owned_by="jina", context_window=512),
        Model(id="jina-reranker-v1-turbo-en", owned_by="jina", context_window=512),
        Model(id="jina-reranker-v1-tiny-en", owned_by="jina", context_window=512),
        Model(id="jina-colbert-v2", owned_by="jina", context_window=8192),
    ]

    # Cache results
    _model_cache.set(cache_key, models)

    return models


def get_voyage_models(
    api_key: Optional[str] = None,
) -> List[Model]:
    """Get available models from Voyage AI.

    Note: Voyage doesn't provide a models API endpoint, so this returns
    a hardcoded list of known models.

    Args:
        api_key: Voyage API key (not used, kept for consistency)

    Returns:
        List of known Voyage models
    """
    # Check cache
    cache_key = _create_cache_key("voyage")
    cached_models = _model_cache.get(cache_key)
    if cached_models is not None:
        return cached_models

    # Hardcoded list of known Voyage models
    models = [
        # Embedding models
        Model(id="voyage-3", owned_by="voyage", context_window=32000),
        Model(id="voyage-3-lite", owned_by="voyage", context_window=32000),
        Model(id="voyage-finance-2", owned_by="voyage", context_window=32000),
        Model(id="voyage-multilingual-2", owned_by="voyage", context_window=32000),
        Model(id="voyage-law-2", owned_by="voyage", context_window=16000),
        Model(id="voyage-code-2", owned_by="voyage", context_window=16000),
        Model(id="voyage-2", owned_by="voyage", context_window=16000),
        Model(id="voyage-large-2", owned_by="voyage", context_window=16000),
        Model(id="voyage-large-2-instruct", owned_by="voyage", context_window=16000),
        # Reranker models
        Model(id="rerank-2", owned_by="voyage", context_window=8000),
        Model(id="rerank-2-lite", owned_by="voyage", context_window=8000),
        Model(id="rerank-1", owned_by="voyage", context_window=8000),
    ]

    # Cache results
    _model_cache.set(cache_key, models)

    return models


def get_vertex_models(
    project_id: Optional[str] = None,
    location: Optional[str] = None,
) -> List[Model]:
    """Get available models from Google Vertex AI.

    Note: Vertex AI uses the same models as Google Gemini but accessed through
    a different endpoint. This returns the same hardcoded list.

    Args:
        project_id: Google Cloud project ID (not used for model listing)
        location: Google Cloud location (not used for model listing)

    Returns:
        List of known Vertex AI models
    """
    # Vertex uses same models as Google Gemini
    # We could call get_google_models but for now return hardcoded list
    cache_key = _create_cache_key("vertex")
    cached_models = _model_cache.get(cache_key)
    if cached_models is not None:
        return cached_models

    models = [
        Model(id="gemini-1.5-pro", owned_by="google", context_window=2097152),
        Model(id="gemini-1.5-flash", owned_by="google", context_window=1048576),
        Model(id="gemini-1.0-pro", owned_by="google", context_window=32760),
        Model(id="gemini-pro", owned_by="google", context_window=32760),
        Model(id="text-embedding-004", owned_by="google", context_window=2048),
        Model(id="text-multilingual-embedding-002", owned_by="google", context_window=2048),
    ]

    _model_cache.set(cache_key, models)
    return models


def get_deepseek_models(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> List[Model]:
    """Get available models from DeepSeek.

    Args:
        api_key: DeepSeek API key (or DEEPSEEK_API_KEY env var)
        base_url: Base URL for API (default: https://api.deepseek.com/v1)

    Returns:
        List of available models

    Raises:
        ValueError: If API key is not provided
        RuntimeError: If API request fails
    """
    # Get API key
    api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("DeepSeek API key not found. Provide api_key or set DEEPSEEK_API_KEY environment variable.")

    # Set defaults
    base_url = base_url or "https://api.deepseek.com/v1"

    # Check cache
    cache_key = _create_cache_key("deepseek", api_key=api_key, base_url=base_url)
    cached_models = _model_cache.get(cache_key)
    if cached_models is not None:
        return cached_models

    # Prepare headers
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # Make request
    try:
        response = httpx.get(
            f"{base_url}/models",
            headers=headers,
            timeout=60.0
        )

        if response.status_code >= 400:
            try:
                error_data = response.json()
                error_message = error_data.get("error", {}).get("message", f"HTTP {response.status_code}")
            except Exception:
                error_message = f"HTTP {response.status_code}: {response.text}"
            raise RuntimeError(f"DeepSeek API error: {error_message}")

        models_data = response.json()

        # Parse models
        all_models = []
        for model in models_data.get("data", []):
            all_models.append(Model(
                id=model["id"],
                owned_by=model.get("owned_by", "deepseek"),
                context_window=model.get("context_window", None),
            ))

        # Cache results
        _model_cache.set(cache_key, all_models)

        return all_models

    except httpx.HTTPError as e:
        raise RuntimeError(f"Failed to fetch DeepSeek models: {e}")


def get_openrouter_models(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> List[Model]:
    """Get available models from OpenRouter.

    Args:
        api_key: OpenRouter API key (or OPENROUTER_API_KEY env var)
        base_url: Base URL for API (default: https://openrouter.ai/api/v1)

    Returns:
        List of available models

    Raises:
        RuntimeError: If API request fails
    """
    # Get API key (optional for OpenRouter model listing)
    api_key = api_key or os.getenv("OPENROUTER_API_KEY")

    # Set defaults
    base_url = base_url or "https://openrouter.ai/api/v1"

    # Check cache
    cache_key = _create_cache_key("openrouter", api_key=api_key or "", base_url=base_url)
    cached_models = _model_cache.get(cache_key)
    if cached_models is not None:
        return cached_models

    # Prepare headers
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    # Make request
    try:
        response = httpx.get(
            f"{base_url}/models",
            headers=headers,
            timeout=60.0
        )

        if response.status_code >= 400:
            try:
                error_data = response.json()
                error_message = error_data.get("error", {}).get("message", f"HTTP {response.status_code}")
            except Exception:
                error_message = f"HTTP {response.status_code}: {response.text}"
            raise RuntimeError(f"OpenRouter API error: {error_message}")

        models_data = response.json()

        # Parse models
        all_models = []
        for model in models_data.get("data", []):
            all_models.append(Model(
                id=model["id"],
                owned_by=model.get("owned_by", "openrouter"),
                context_window=model.get("context_length", None),
            ))

        # Cache results
        _model_cache.set(cache_key, all_models)

        return all_models

    except httpx.HTTPError as e:
        raise RuntimeError(f"Failed to fetch OpenRouter models: {e}")


def get_xai_models(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> List[Model]:
    """Get available models from xAI.

    Args:
        api_key: xAI API key (or XAI_API_KEY env var)
        base_url: Base URL for API (default: https://api.x.ai/v1)

    Returns:
        List of available models

    Raises:
        ValueError: If API key is not provided
        RuntimeError: If API request fails
    """
    # Get API key
    api_key = api_key or os.getenv("XAI_API_KEY")
    if not api_key:
        raise ValueError("xAI API key not found. Provide api_key or set XAI_API_KEY environment variable.")

    # Set defaults
    base_url = base_url or "https://api.x.ai/v1"

    # Check cache
    cache_key = _create_cache_key("xai", api_key=api_key, base_url=base_url)
    cached_models = _model_cache.get(cache_key)
    if cached_models is not None:
        return cached_models

    # Prepare headers
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # Make request
    try:
        response = httpx.get(
            f"{base_url}/models",
            headers=headers,
            timeout=60.0
        )

        if response.status_code >= 400:
            try:
                error_data = response.json()
                error_message = error_data.get("error", {}).get("message", f"HTTP {response.status_code}")
            except Exception:
                error_message = f"HTTP {response.status_code}: {response.text}"
            raise RuntimeError(f"xAI API error: {error_message}")

        models_data = response.json()

        # Parse models
        all_models = []
        for model in models_data.get("data", []):
            all_models.append(Model(
                id=model["id"],
                owned_by=model.get("owned_by", "xai"),
                context_window=model.get("context_window", None),
            ))

        # Cache results
        _model_cache.set(cache_key, all_models)

        return all_models

    except httpx.HTTPError as e:
        raise RuntimeError(f"Failed to fetch xAI models: {e}")


def get_perplexity_models(
    api_key: Optional[str] = None,
) -> List[Model]:
    """Get available models from Perplexity.

    Note: Perplexity doesn't provide a models API endpoint, so this returns
    a hardcoded list of known models.

    Args:
        api_key: Perplexity API key (not used, kept for consistency)

    Returns:
        List of known Perplexity models
    """
    # Check cache
    cache_key = _create_cache_key("perplexity")
    cached_models = _model_cache.get(cache_key)
    if cached_models is not None:
        return cached_models

    # Hardcoded list of known Perplexity models
    models = [
        Model(id="llama-3.1-sonar-small-128k-online", owned_by="perplexity", context_window=127072),
        Model(id="llama-3.1-sonar-large-128k-online", owned_by="perplexity", context_window=127072),
        Model(id="llama-3.1-sonar-huge-128k-online", owned_by="perplexity", context_window=127072),
        Model(id="llama-3.1-sonar-small-128k-chat", owned_by="perplexity", context_window=131072),
        Model(id="llama-3.1-sonar-large-128k-chat", owned_by="perplexity", context_window=131072),
        Model(id="llama-3.1-8b-instruct", owned_by="perplexity", context_window=131072),
        Model(id="llama-3.1-70b-instruct", owned_by="perplexity", context_window=131072),
    ]

    # Cache results
    _model_cache.set(cache_key, models)

    return models


def get_azure_models(
    api_key: Optional[str] = None,
    azure_endpoint: Optional[str] = None,
    api_version: Optional[str] = None,
) -> List[Model]:
    """Get available models from Azure OpenAI.

    Note: Azure OpenAI uses deployments, not a models API. This returns
    an empty list as models are deployment-specific.

    Args:
        api_key: Azure OpenAI API key
        azure_endpoint: Azure OpenAI endpoint
        api_version: Azure API version

    Returns:
        Empty list (Azure uses deployments, not discoverable models)
    """
    # Azure doesn't have a models API - users create deployments
    # Return empty list
    return []


def get_openai_compatible_models(
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    model_type: Optional[str] = None,
) -> List[Model]:
    """Get available models from OpenAI-compatible endpoints.

    This function attempts to fetch models from providers that implement the
    OpenAI API specification (e.g., LM Studio, vLLM, custom endpoints).

    Args:
        base_url: Base URL for the API endpoint (required)
        api_key: API key if required by the endpoint
        model_type: Type of models to return (optional filtering)

    Returns:
        List of available models

    Raises:
        ValueError: If base_url is not provided
        RuntimeError: If API request fails
    """
    if not base_url:
        raise ValueError(
            "base_url is required for OpenAI-compatible model discovery. "
            "Provide the base URL of your OpenAI-compatible endpoint "
            "(e.g., 'http://localhost:1234/v1' for LM Studio)."
        )

    # Check cache
    cache_key = _create_cache_key("openai-compatible", base_url=base_url, model_type=model_type)
    cached_models = _model_cache.get(cache_key)
    if cached_models is not None:
        return cached_models

    # Prepare headers
    headers = {
        "Content-Type": "application/json",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    # Make request
    try:
        # Ensure base_url doesn't have trailing slash
        base_url = base_url.rstrip("/")

        response = httpx.get(
            f"{base_url}/models",
            headers=headers,
            timeout=60.0
        )

        if response.status_code >= 400:
            try:
                error_data = response.json()
                error_message = error_data.get("error", {}).get("message", f"HTTP {response.status_code}")
            except Exception:
                error_message = f"HTTP {response.status_code}: {response.text}"
            raise RuntimeError(f"OpenAI-compatible API error: {error_message}")

        models_data = response.json()

        # Parse models - try to handle both OpenAI format and variations
        all_models = []
        models_list = models_data.get("data", [])

        for model in models_list:
            model_id = model.get("id", "")

            # Optional type filtering (if provided)
            if model_type is not None:
                include = False
                if model_type == "language" and ("gpt" in model_id.lower() or "chat" in model_id.lower() or "instruct" in model_id.lower()):
                    include = True
                elif model_type == "embedding" and "embedding" in model_id.lower():
                    include = True
                elif model_type == "speech_to_text" and "whisper" in model_id.lower():
                    include = True
                elif model_type == "text_to_speech" and "tts" in model_id.lower():
                    include = True

                if not include:
                    continue

            all_models.append(Model(
                id=model_id,
                owned_by=model.get("owned_by", "unknown"),
                context_window=model.get("context_window", None),
            ))

        # Cache results
        _model_cache.set(cache_key, all_models)

        return all_models

    except httpx.HTTPError as e:
        raise RuntimeError(f"Failed to fetch OpenAI-compatible models: {e}")


def get_transformers_models(
    cache_dir: Optional[str] = None,
) -> List[Model]:
    """Get available local Transformers models.

    Note: This would require scanning the local cache directory for downloaded
    models. For now, returns an empty list.

    Args:
        cache_dir: Optional cache directory to scan

    Returns:
        List of locally available models (currently empty)
    """
    # Would need to scan cache directory and detect model types
    # This is complex and provider-specific, so return empty list for now
    return []


# Provider registry mapping provider names to discovery functions
PROVIDER_MODELS_REGISTRY: Dict[str, Callable[..., List[Model]]] = {
    "openai": get_openai_models,
    "openai-compatible": get_openai_compatible_models,
    "anthropic": get_anthropic_models,
    "google": get_google_models,
    "vertex": get_vertex_models,
    "mistral": get_mistral_models,
    "groq": get_groq_models,
    "deepseek": get_deepseek_models,
    "ollama": get_ollama_models,
    "openrouter": get_openrouter_models,
    "xai": get_xai_models,
    "perplexity": get_perplexity_models,
    "jina": get_jina_models,
    "voyage": get_voyage_models,
    "azure": get_azure_models,
    "transformers": get_transformers_models,
}
