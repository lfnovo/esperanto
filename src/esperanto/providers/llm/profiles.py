"""OpenAI-compatible provider profiles for config-driven virtual providers."""

import warnings
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Dict, Literal, Optional, Set, get_args

Modality = Literal["language", "embedding", "speech_to_text", "text_to_speech"]
"""The modalities a profile can declare support for."""

_VALID_MODALITIES: Set[str] = set(get_args(Modality))


@dataclass(frozen=True)
class OpenAICompatibleProfile:
    """Declarative configuration for an OpenAI-compatible provider.

    Instead of creating a new Python class for each OpenAI-compatible endpoint,
    define a profile and register it. The factory creates the appropriate
    ``OpenAICompatible*Model`` for each modality the profile declares.

    A profile only serves the modalities listed in ``capabilities`` (default:
    ``{"language"}``). Requesting a modality the profile does not declare raises
    ``ProviderCapabilityError``. This keeps the provider matrix honest — a
    chat-only endpoint never advertises itself as an embedding provider.

    Example:
        >>> from esperanto import AIFactory
        >>> from esperanto.providers.llm.profiles import OpenAICompatibleProfile
        >>>
        >>> AIFactory.register_openai_compatible_profile(
        ...     OpenAICompatibleProfile(
        ...         name="together",
        ...         base_url="https://api.together.xyz/v1",
        ...         api_key_env="TOGETHER_API_KEY",
        ...         default_models={"language": "meta-llama/Llama-3-70b-chat-hf"},
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

    default_model: Optional[str] = None
    """Deprecated. Back-compat alias for ``default_models['language']``. Kept in
    its original positional slot so existing positional construction still binds
    here. Passing this emits a DeprecationWarning; use ``default_models`` instead."""

    capabilities: Set[Modality] = field(default_factory=lambda: {"language"})
    """Modalities this profile serves. Defaults to language-only."""

    default_models: Dict[Modality, str] = field(default_factory=dict)
    """Default model per modality when the caller passes none. A modality with
    no entry has no default — the caller must pass a model name or a clear error
    is raised (never a generic placeholder)."""

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

    requires_api_key: bool = True
    """Whether the endpoint requires an API key. Set False for local/no-auth
    endpoints (e.g. oMLX): a missing key then falls back to 'not-required'
    instead of raising, mirroring the generic openai-compatible path.
    (Kept last so new fields don't shift existing positional slots.)"""

    def __post_init__(self) -> None:
        # Validate declared capabilities against the known modalities, so a typo
        # can't leak an invalid category into get_available_providers().
        invalid = sorted(c for c in self.capabilities if c not in _VALID_MODALITIES)
        if invalid:
            raise ValueError(
                f"Unknown profile capabilities {invalid}; "
                f"valid modalities are {sorted(_VALID_MODALITIES)}."
            )

        # Fold the deprecated ``default_model`` into ``default_models['language']``.
        models = dict(self.default_models)
        if self.default_model is not None:
            warnings.warn(
                "OpenAICompatibleProfile(default_model=...) is deprecated; "
                'use default_models={"language": ...} instead.',
                DeprecationWarning,
                stacklevel=3,
            )
            models.setdefault("language", self.default_model)

        # Freeze the mutable collections. The dataclass is frozen (no rebinding),
        # but a Set/Dict field is still mutable in place — and built-in profiles
        # are shared global state, so a stray .add()/[...]= could drift behavior
        # for every caller. Store immutable copies instead.
        object.__setattr__(self, "capabilities", frozenset(self.capabilities))
        object.__setattr__(self, "default_models", MappingProxyType(models))

    def default_model_for(self, modality: Modality) -> Optional[str]:
        """Return the default model for a modality, or None if none is set."""
        return self.default_models.get(modality)


BUILTIN_PROFILES: Dict[str, OpenAICompatibleProfile] = {
    "deepseek": OpenAICompatibleProfile(
        name="deepseek",
        base_url="https://api.deepseek.com/v1",
        api_key_env="DEEPSEEK_API_KEY",
        base_url_env="DEEPSEEK_BASE_URL",
        default_models={"language": "deepseek-chat"},
        owned_by="DeepSeek",
        display_name="DeepSeek",
    ),
    "xai": OpenAICompatibleProfile(
        name="xai",
        base_url="https://api.x.ai/v1",
        api_key_env="XAI_API_KEY",
        base_url_env="XAI_BASE_URL",
        default_models={"language": "grok-2-latest"},
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
        default_models={"language": "qwen-plus"},
        owned_by="Alibaba Cloud",
        display_name="DashScope",
    ),
    "minimax": OpenAICompatibleProfile(
        name="minimax",
        base_url="https://api.minimax.io/v1",
        api_key_env="MINIMAX_API_KEY",
        base_url_env="MINIMAX_BASE_URL",
        default_models={"language": "MiniMax-M2.5"},
        owned_by="MiniMax",
        display_name="MiniMax",
    ),
    "novita": OpenAICompatibleProfile(
        name="novita",
        base_url="https://api.novita.ai/openai",
        api_key_env="NOVITA_API_KEY",
        base_url_env="NOVITA_BASE_URL",
        default_models={"language": "moonshotai/kimi-k2.5"},
        owned_by="Novita",
        display_name="Novita",
    ),
    "ppq": OpenAICompatibleProfile(
        name="ppq",
        base_url="https://api.ppq.ai/v1",
        api_key_env="PPQ_API_KEY",
        base_url_env="PPQ_BASE_URL",
        capabilities={"language", "embedding", "speech_to_text", "text_to_speech"},
        default_models={
            "language": "auto",
            "embedding": "openai/text-embedding-3-small",
            "speech_to_text": "nova-3",
            "text_to_speech": "deepgram_aura_2",
        },
        owned_by="PayPerQ",
        display_name="PayPerQ",
    ),
    "omlx": OpenAICompatibleProfile(
        name="omlx",
        base_url="http://localhost:11435/v1",
        api_key_env="OMLX_API_KEY",
        base_url_env="OMLX_API_BASE",
        capabilities={"language", "embedding"},
        default_models={},  # bring-your-own-models: no fixed default
        requires_api_key=False,
        owned_by="oMLX",
        display_name="oMLX",
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


def get_profile_capabilities() -> Dict[str, Set[Modality]]:
    """Map every known profile name to its declared capabilities.

    User profiles take precedence over builtins of the same name.
    """
    merged: Dict[str, OpenAICompatibleProfile] = {**BUILTIN_PROFILES, **_USER_PROFILES}
    return {name: set(profile.capabilities) for name, profile in merged.items()}
