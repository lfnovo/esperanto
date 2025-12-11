"""Simple JSONL-based metrics logger for LLM performance tracking."""

from __future__ import annotations

import json
import os
import platform
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


def _get_default_metrics_path() -> str:
    """
    Get platform-appropriate default metrics path.

    Respects BRIODOCS_ENV for dev/prod isolation:
    - BRIODOCS_ENV=dev -> logs-dev/metrics.jsonl
    - BRIODOCS_ENV=prod (default) -> logs/metrics.jsonl

    Platform locations:
    - macOS: ~/Library/Application Support/BrioDocs/
    - Windows: %APPDATA%/BrioDocs/
    - Linux: ~/.config/briodocs/
    """
    system = platform.system()
    env = os.getenv("BRIODOCS_ENV", "prod").lower()
    suffix = "-dev" if env == "dev" else ""

    if system == "Darwin":
        base = Path.home() / "Library" / "Application Support" / "BrioDocs"
    elif system == "Windows":
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming")) / "BrioDocs"
    else:
        base = Path.home() / ".config" / "briodocs"

    return str(base / f"logs{suffix}" / "metrics.jsonl")


class MetricsLogger:
    """
    Logs LLM performance metrics to a JSONL file.

    Each log entry includes:
    - Timestamp
    - Tier information (id, label, context_size)
    - Model name
    - Timing metrics (TTFT, tokens/second)
    - Usage data (token counts)

    Example log entry:
        {"ts": "2025-12-08T10:30:00Z", "tier_id": "fast", "tier_label": "Fast",
         "model": "llama-3.2-3b-q4", "context_size": 4096, "ttft_ms": 145,
         "tokens_per_second": 23.5, "prompt_tokens": 88, "completion_tokens": 42}
    """

    def __init__(self, log_path: Optional[Path] = None):
        """
        Initialize the metrics logger.

        Args:
            log_path: Path to the JSONL log file.
                      Defaults to platform-appropriate location with env-based isolation.
                      Can be overridden with BRIODOCS_METRICS_PATH env var.
        """
        if log_path is None:
            log_path = Path(os.getenv("BRIODOCS_METRICS_PATH", _get_default_metrics_path()))
        self.log_path = Path(log_path).expanduser()
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(
        self,
        tier_id: str,
        model: str,
        tier_label: Optional[str] = None,
        context_size: Optional[int] = None,
        ttft_ms: Optional[float] = None,
        tokens_per_second: Optional[float] = None,
        prompt_tokens_per_second: Optional[float] = None,
        total_time_ms: Optional[float] = None,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
        total_tokens: Optional[int] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Log a metrics record.

        Args:
            tier_id: Unique identifier for the tier (e.g., "fast", "quality")
            model: Model name used for the request
            tier_label: Human-readable tier name (optional)
            context_size: Context window size for this tier (optional)
            ttft_ms: Time to first token in milliseconds (optional)
            tokens_per_second: Token generation speed (optional)
            prompt_tokens_per_second: Prompt processing speed (optional)
            total_time_ms: Total request time in milliseconds (optional)
            prompt_tokens: Number of prompt tokens (optional)
            completion_tokens: Number of completion tokens (optional)
            total_tokens: Total tokens used (optional)
            extra: Additional custom fields to include (optional)

        Returns:
            The logged record as a dictionary
        """
        record: Dict[str, Any] = {
            "ts": datetime.utcnow().isoformat() + "Z",
            "tier_id": tier_id,
            "model": model,
        }

        # Add optional fields only if provided
        if tier_label is not None:
            record["tier_label"] = tier_label
        if context_size is not None:
            record["context_size"] = context_size
        if ttft_ms is not None:
            record["ttft_ms"] = round(ttft_ms, 2)
        if tokens_per_second is not None:
            record["tokens_per_second"] = round(tokens_per_second, 2)
        if prompt_tokens_per_second is not None:
            record["prompt_tokens_per_second"] = round(prompt_tokens_per_second, 2)
        if total_time_ms is not None:
            record["total_time_ms"] = round(total_time_ms, 2)
        if prompt_tokens is not None:
            record["prompt_tokens"] = prompt_tokens
        if completion_tokens is not None:
            record["completion_tokens"] = completion_tokens
        if total_tokens is not None:
            record["total_tokens"] = total_tokens

        # Merge any extra fields
        if extra:
            record.update(extra)

        # Append to log file
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        return record

    def log_from_response(
        self,
        tier_id: str,
        model: str,
        timings: Optional[Dict[str, Any]] = None,
        usage: Optional[Dict[str, Any]] = None,
        tier_label: Optional[str] = None,
        context_size: Optional[int] = None,
        ttft_ms: Optional[float] = None,
        request_time_ms: Optional[float] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Log metrics from a ChatCompletion response.

        Convenience method that extracts fields from timings and usage dicts.

        Args:
            tier_id: Unique identifier for the tier
            model: Model name used for the request
            timings: Timings dict from response (e.g., response.timings.model_dump())
            usage: Usage dict from response (e.g., response.usage.model_dump())
            tier_label: Human-readable tier name (optional)
            context_size: Context window size for this tier (optional)
            ttft_ms: Override TTFT if measured externally (optional)
            request_time_ms: Wall-clock request time in milliseconds (optional)
            extra: Additional custom fields to include (optional)

        Returns:
            The logged record as a dictionary
        """
        timings = timings or {}
        usage = usage or {}

        # Get tokens_per_second from llama.cpp timings, or calculate from request time
        tokens_per_second = timings.get("tokens_per_second")
        completion_tokens = usage.get("completion_tokens")

        # Calculate tokens_per_second if not provided but we have the data
        if tokens_per_second is None and request_time_ms and completion_tokens:
            request_time_sec = request_time_ms / 1000
            if request_time_sec > 0:
                tokens_per_second = completion_tokens / request_time_sec

        return self.log(
            tier_id=tier_id,
            model=model,
            tier_label=tier_label,
            context_size=context_size,
            ttft_ms=ttft_ms or timings.get("ttft_ms"),
            tokens_per_second=tokens_per_second,
            prompt_tokens_per_second=timings.get("prompt_tokens_per_second"),
            total_time_ms=request_time_ms or timings.get("total_time_ms"),
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=completion_tokens,
            total_tokens=usage.get("total_tokens"),
            extra=extra,
        )

    def get_recent(self, n: int = 100) -> List[Dict[str, Any]]:
        """
        Read the last n metrics records.

        Args:
            n: Maximum number of records to return (default 100)

        Returns:
            List of metric records, most recent last
        """
        if not self.log_path.exists():
            return []

        records = []
        with open(self.log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

        return records[-n:]

    def get_stats(self, tier_id: Optional[str] = None, model: Optional[str] = None) -> Dict[str, Any]:
        """
        Calculate summary statistics for logged metrics.

        Args:
            tier_id: Filter by tier (optional)
            model: Filter by model (optional)

        Returns:
            Dictionary with count, averages, and ranges for key metrics
        """
        records = self.get_recent(1000)  # Analyze last 1000 records

        # Filter if requested
        if tier_id:
            records = [r for r in records if r.get("tier_id") == tier_id]
        if model:
            records = [r for r in records if r.get("model") == model]

        if not records:
            return {"count": 0}

        # Collect numeric values
        ttft_values = [r["ttft_ms"] for r in records if "ttft_ms" in r]
        tps_values = [r["tokens_per_second"] for r in records if "tokens_per_second" in r]
        completion_values = [r["completion_tokens"] for r in records if "completion_tokens" in r]

        stats: Dict[str, Any] = {"count": len(records)}

        if ttft_values:
            stats["ttft_ms"] = {
                "avg": round(sum(ttft_values) / len(ttft_values), 2),
                "min": round(min(ttft_values), 2),
                "max": round(max(ttft_values), 2),
            }

        if tps_values:
            stats["tokens_per_second"] = {
                "avg": round(sum(tps_values) / len(tps_values), 2),
                "min": round(min(tps_values), 2),
                "max": round(max(tps_values), 2),
            }

        if completion_values:
            stats["completion_tokens"] = {
                "avg": round(sum(completion_values) / len(completion_values), 1),
                "total": sum(completion_values),
            }

        return stats

    def clear(self) -> None:
        """Clear all logged metrics."""
        if self.log_path.exists():
            self.log_path.unlink()
