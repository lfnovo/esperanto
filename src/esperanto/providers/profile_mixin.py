"""Shared OpenAI-compatible profile resolution across all modalities.

The four ``OpenAICompatible*Model`` classes (language, embedding, STT, TTS) each
need to resolve ``base_url`` / ``api_key`` / default model from the same
precedence chain, optionally driven by a registered provider profile. Keeping
that chain in one place is the whole point of issue #230 — a copy per modality
is exactly the kind of silent divergence this feature exists to remove.
"""

import os
from typing import Any, Dict, Optional

from esperanto.providers.llm.profiles import (
    Modality,
    OpenAICompatibleProfile,
    get_profile,
)

# Modality -> suffix used in the generic OPENAI_COMPATIBLE_* env vars.
_ENV_SUFFIX: Dict[str, str] = {
    "language": "LLM",
    "embedding": "EMBEDDING",
    "speech_to_text": "STT",
    "text_to_speech": "TTS",
}


class ProfileAwareMixin:
    """Resolves configuration for OpenAI-compatible providers, profile-aware.

    When a profile is active (``config['_profile_name']`` set), configuration
    flows from the profile; otherwise it flows from the generic
    ``OPENAI_COMPATIBLE_*`` env vars. Either way the precedence order is:
    explicit param > config dict > env var > profile/hardcoded default.
    """

    def _resolve_profile(
        self, config: Dict[str, Any], modality: Modality
    ) -> Optional[OpenAICompatibleProfile]:
        """Look up the active profile from config, or None if not profile-driven.

        Enforces the modality capability at the construction boundary too, not
        just in the factory — so directly constructing a modality adapter with a
        profile that doesn't declare that modality fails fast rather than reaching
        the wrong endpoint.
        """
        profile_name = config.get("_profile_name")
        if not profile_name:
            return None
        profile = get_profile(profile_name)
        if not profile:
            raise ValueError(f"Unknown provider profile: '{profile_name}'")
        if modality not in profile.capabilities:
            from esperanto.common_types import ProviderCapabilityError

            display = profile.display_name or profile.name
            raise ProviderCapabilityError(
                f"Provider '{profile.name}' ({display}) does not support {modality}. "
                f"Declared capabilities: {sorted(profile.capabilities)}."
            )
        return profile

    def _resolve_base_url(
        self,
        modality: Modality,
        profile: Optional[OpenAICompatibleProfile],
        explicit: Optional[str],
        config: Dict[str, Any],
    ) -> Optional[str]:
        if profile:
            return (
                explicit
                or config.get("base_url")
                or os.getenv(profile.base_url_env or "")
                or profile.base_url
            )
        suffix = _ENV_SUFFIX[modality]
        return (
            explicit
            or config.get("base_url")
            or os.getenv(f"OPENAI_COMPATIBLE_BASE_URL_{suffix}")
            or os.getenv("OPENAI_COMPATIBLE_BASE_URL")
        )

    def _resolve_api_key(
        self,
        modality: Modality,
        profile: Optional[OpenAICompatibleProfile],
        explicit: Optional[str],
        config: Dict[str, Any],
    ) -> Optional[str]:
        if profile:
            return explicit or config.get("api_key") or os.getenv(profile.api_key_env)
        suffix = _ENV_SUFFIX[modality]
        return (
            explicit
            or config.get("api_key")
            or os.getenv(f"OPENAI_COMPATIBLE_API_KEY_{suffix}")
            or os.getenv("OPENAI_COMPATIBLE_API_KEY")
        )

    def _resolve_default_model(
        self,
        modality: Modality,
        profile: Optional[OpenAICompatibleProfile],
        generic_default: str,
    ) -> str:
        """Return the default model for this modality.

        With no profile, fall back to the endpoint-generic default. With a
        profile that declares this modality but sets no default (a
        bring-your-own-models server), raise rather than return a placeholder
        the endpoint has never heard of — no default beats a wrong default.
        """
        if profile is None:
            return generic_default
        model = profile.default_model_for(modality)
        if model:
            return model
        display = profile.display_name or profile.name
        raise ValueError(
            f"{display} profile declares '{modality}' but sets no default model. "
            f"Pass model_name explicitly."
        )
