"""DeepSeek language model implementation.

Deprecated: Use AIFactory.create_language('deepseek', ...) instead.
DeepSeek is now handled as an OpenAI-compatible provider profile.
"""

import warnings
from dataclasses import dataclass

from esperanto.providers.llm.openai_compatible import OpenAICompatibleLanguageModel


@dataclass
class DeepSeekLanguageModel(OpenAICompatibleLanguageModel):
    """DeepSeek language model.

    .. deprecated::
        Use ``AIFactory.create_language('deepseek', ...)`` instead.
        This class is kept for backwards compatibility.
    """

    model_name: str = "deepseek-chat"

    def __post_init__(self):
        warnings.warn(
            "DeepSeekLanguageModel is deprecated. "
            "Use AIFactory.create_language('deepseek', ...) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if not self.config:
            self.config = {}
        self.config["_profile_name"] = "deepseek"
        super().__post_init__()
