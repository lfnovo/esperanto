"""OpenAI-compatible provider profiles for config-driven virtual providers."""

from dataclasses import dataclass
from typing import Dict, Optional, Set


@dataclass(frozen=True)
class OpenAICompatibleProfile:
    """Declarative configuration for an OpenAI-compatible provider.

    Instead of creating a new Python class for each OpenAI-compatible endpoint,
    define a profile and register it. The factory will create an
    OpenAICompatibleLanguageModel configured by the profile.

    Example:
        >>> from esperanto import AIFactory
        >>> from esperanto.providers.llm.profiles import OpenAICompatibleProfile
        >>>
        >>> AIFactory.register_openai_compatible_profile(
        ...     OpenAICompatibleProfile(
        ...         name="together",
        ...         base_url="https://api.together.xyz/v1",
        ...         api_key_env="TOGETHER_API_KEY",
        ...         default_model="meta-llama/Llama-3-70b-chat-hf",
        ...     )
        ... )
        >>> model = AIFactory.create_language("together", "meta-llama/Llama-3-70b-chat-hf")
    """

    name: str
    """Provider identifier used in AIFactory.create_language(provider=...)."""

    base_url: str
    """Default API endpoint URL."""

    api_key_env: str
    """Environment variable name for the API key (e.g., 'DEEPSEEK_API_KEY')."""

    default_model: str
    """Default model name when none is specified."""

    base_url_env: Optional[str] = None
    """Optional environment variable for base URL override."""

    supports_response_format: bool = True
    """Whether the endpoint supports the response_format parameter."""

    model_prefix_filter: Optional[str] = None
    """Only include models whose ID starts with this prefix in _get_models()."""

    owned_by: Optional[str] = None
    """Override the owned_by field in model listings."""

    display_name: Optional[str] = None
    """Human-readable name for error messages (e.g., 'DeepSeek'). Defaults to name."""


BUILTIN_PROFILES: Dict[str, OpenAICompatibleProfile] = {
    "deepseek": OpenAICompatibleProfile(
        name="deepseek",
        base_url="https://api.deepseek.com/v1",
        api_key_env="DEEPSEEK_API_KEY",
        base_url_env="DEEPSEEK_BASE_URL",
        default_model="deepseek-chat",
        owned_by="DeepSeek",
        display_name="DeepSeek",
    ),
    "xai": OpenAICompatibleProfile(
        name="xai",
        base_url="https://api.x.ai/v1",
        api_key_env="XAI_API_KEY",
        base_url_env="XAI_BASE_URL",
        default_model="grok-2-latest",
        supports_response_format=False,
        model_prefix_filter="grok",
        owned_by="X.AI",
        display_name="XAI",
    ),
    "dashscope": OpenAICompatibleProfile(
        name="dashscope",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key_env="DASHSCOPE_API_KEY",
        base_url_env="DASHSCOPE_BASE_URL",
        default_model="qwen-plus",
        owned_by="Alibaba Cloud",
        display_name="DashScope",
    ),
    "minimax": OpenAICompatibleProfile(
        name="minimax",
        base_url="https://api.minimax.io/v1",
        api_key_env="MINIMAX_API_KEY",
        base_url_env="MINIMAX_BASE_URL",
        default_model="MiniMax-M2.5",
        owned_by="MiniMax",
        display_name="MiniMax",
    ),
}

_USER_PROFILES: Dict[str, OpenAICompatibleProfile] = {}


def get_profile(name: str) -> Optional[OpenAICompatibleProfile]:
    """Look up a profile by name. User profiles take precedence over builtins."""
    return _USER_PROFILES.get(name) or BUILTIN_PROFILES.get(name)


def register_profile(profile: OpenAICompatibleProfile) -> None:
    """Register a user-defined provider profile."""
    if not isinstance(profile, OpenAICompatibleProfile):
        raise TypeError("profile must be an OpenAICompatibleProfile instance")
    if not profile.name:
        raise ValueError("profile.name must be a non-empty string")
    normalized_name = profile.name.lower().replace("_", "-")
    _USER_PROFILES[normalized_name] = profile


def get_all_profile_names() -> Set[str]:
    """Return the combined set of builtin and user profile names."""
    return set(BUILTIN_PROFILES.keys()) | set(_USER_PROFILES.keys())
