"""XAI language model implementation.

Deprecated: Use AIFactory.create_language('xai', ...) instead.
XAI is now handled as an OpenAI-compatible provider profile.
"""

import warnings
from dataclasses import dataclass

from esperanto.providers.llm.openai_compatible import OpenAICompatibleLanguageModel


@dataclass
class XAILanguageModel(OpenAICompatibleLanguageModel):
    """XAI (Grok) language model.

    .. deprecated::
        Use ``AIFactory.create_language('xai', ...)`` instead.
        This class is kept for backwards compatibility.
    """

    def __post_init__(self):
        warnings.warn(
            "XAILanguageModel is deprecated. "
            "Use AIFactory.create_language('xai', ...) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if not self.config:
            self.config = {}
        self.config["_profile_name"] = "xai"
        super().__post_init__()
