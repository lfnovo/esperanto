"""Helpers for schema-driven structured outputs."""

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type, Union

from pydantic import BaseModel, ValidationError

from esperanto.common_types.exceptions import StructuredOutputValidationError


@dataclass(frozen=True)
class ResolvedStructuredOutput:
    """Normalized structured output configuration."""

    mode: str
    response_format: Dict[str, Any]
    schema_source: Optional[Union[Type[BaseModel], Dict[str, Any]]] = None
    schema_name: Optional[str] = None

    @property
    def is_schema_mode(self) -> bool:
        """Whether this config enforces a JSON schema."""
        return self.mode == "json_schema"


def _is_pydantic_model_class(value: Any) -> bool:
    """Check whether a value is a Pydantic model class."""
    return isinstance(value, type) and issubclass(value, BaseModel)


def _validate_schema_name(name: Any) -> str:
    """Validate schema name and return normalized value."""
    if not isinstance(name, str):
        raise ValueError("structured['name'] must be a non-empty string")
    normalized = name.strip()
    if not normalized:
        raise ValueError("structured['name'] must be a non-empty string")

    # OpenAI-compatible schema name constraints.
    if len(normalized) > 64:
        raise ValueError("structured['name'] must be at most 64 characters")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", normalized):
        raise ValueError(
            "structured['name'] may only contain letters, digits, underscores, and dashes"
        )
    return normalized


def _validate_strict(strict: Any) -> bool:
    """Validate strict flag for schema mode."""
    if not isinstance(strict, bool):
        raise TypeError("structured['strict'] must be a boolean")
    return strict


def resolve_structured_output(
    structured: Any,
    *,
    allow_string_json_alias: bool = False,
) -> Optional[ResolvedStructuredOutput]:
    """Normalize structured output configuration.

    Supports:
      - {"type": "json"} / {"type": "json_object"}
      - {"type": "json_schema", "schema": <Pydantic model class | JSON schema dict>}
      - "json" (optional alias, when allow_string_json_alias=True)
    """
    if structured is None:
        return None

    if isinstance(structured, str):
        if allow_string_json_alias and structured == "json":
            return ResolvedStructuredOutput(
                mode="json_object",
                response_format={"type": "json_object"},
            )
        raise TypeError("structured parameter must be a dictionary")

    if not isinstance(structured, dict):
        raise TypeError("structured parameter must be a dictionary")

    structured_type = structured.get("type")
    if structured_type in ("json", "json_object"):
        return ResolvedStructuredOutput(
            mode="json_object",
            response_format={"type": "json_object"},
        )

    if structured_type != "json_schema":
        raise TypeError(
            "Invalid 'type' in structured dictionary. "
            "Expected 'json', 'json_object', or 'json_schema'."
        )

    if "schema" not in structured:
        raise ValueError("structured['schema'] is required when type='json_schema'")

    schema = structured["schema"]
    strict = _validate_strict(structured.get("strict", True))

    if _is_pydantic_model_class(schema):
        schema_name = structured.get("name", schema.__name__)
        schema_name = _validate_schema_name(schema_name)
        schema_dict = schema.model_json_schema()
        return ResolvedStructuredOutput(
            mode="json_schema",
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "schema": schema_dict,
                    "strict": strict,
                },
            },
            schema_source=schema,
            schema_name=schema_name,
        )

    if isinstance(schema, dict):
        schema_name = structured.get("name", "structured_output")
        schema_name = _validate_schema_name(schema_name)
        return ResolvedStructuredOutput(
            mode="json_schema",
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "schema": schema,
                    "strict": strict,
                },
            },
            schema_source=schema,
            schema_name=schema_name,
        )

    raise TypeError(
        "structured['schema'] must be a Pydantic model class or JSON schema dictionary"
    )


def _format_jsonschema_path(path: Any) -> str:
    """Format a jsonschema error path into a readable JSON path."""
    if not path:
        return "$"
    parts = ["$"]
    for segment in path:
        if isinstance(segment, int):
            parts.append(f"[{segment}]")
        else:
            parts.append(f".{segment}")
    return "".join(parts)


def parse_structured_output_content(
    content: Optional[str],
    resolved: ResolvedStructuredOutput,
) -> Any:
    """Parse and validate model output content for schema mode."""
    if not resolved.is_schema_mode:
        return None

    schema_name = resolved.schema_name or "structured_output"
    raw_content = content or ""

    try:
        parsed = json.loads(raw_content)
    except json.JSONDecodeError as exc:
        raise StructuredOutputValidationError(
            schema_name=schema_name,
            errors=[f"Response content is not valid JSON: {exc.msg}"],
        ) from exc

    schema_source = resolved.schema_source
    if _is_pydantic_model_class(schema_source):
        try:
            return schema_source.model_validate(parsed)
        except ValidationError as exc:
            errors = [str(err) for err in exc.errors()]
            raise StructuredOutputValidationError(
                schema_name=schema_name,
                errors=errors or ["Pydantic validation failed"],
            ) from exc

    if isinstance(schema_source, dict):
        try:
            import jsonschema
        except ImportError:
            # Optional validation dependency is not installed.
            return parsed

        try:
            validator_cls = jsonschema.validators.validator_for(schema_source)
            validator_cls.check_schema(schema_source)
            validator = validator_cls(schema_source)
            validation_errors = sorted(
                validator.iter_errors(parsed),
                key=lambda err: list(err.absolute_path),
            )
        except jsonschema.exceptions.SchemaError as exc:
            raise StructuredOutputValidationError(
                schema_name=schema_name,
                errors=[f"Invalid JSON schema configuration: {exc.message}"],
            ) from exc

        if validation_errors:
            errors = [
                f"{_format_jsonschema_path(err.absolute_path)}: {err.message}"
                for err in validation_errors
            ]
            raise StructuredOutputValidationError(schema_name=schema_name, errors=errors)

        return parsed

    raise StructuredOutputValidationError(
        schema_name=schema_name,
        errors=["Unsupported schema configuration for parsing"],
    )
