"""
Esperanto integrations with third-party frameworks.

This module provides adapters for using Esperanto models with popular
AI/ML frameworks like Pydantic AI.
"""

# Pydantic AI integration - optional import
try:
    from esperanto.integrations.pydantic_ai import EsperantoPydanticModel
except ImportError:
    EsperantoPydanticModel = None  # type: ignore[misc, assignment]

__all__ = ["EsperantoPydanticModel"]
