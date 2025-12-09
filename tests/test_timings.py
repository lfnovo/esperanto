"""Tests for the Timings type and metrics functionality."""

import json
import tempfile
from pathlib import Path

import pytest

from esperanto.common_types import ChatCompletion, Choice, Message, Timings, Usage


class TestTimingsType:
    """Tests for the Timings pydantic model."""

    def test_timings_all_fields(self):
        """Test creating Timings with all fields populated."""
        timings = Timings(
            ttft_ms=150.5,
            tokens_per_second=23.4,
            prompt_tokens_per_second=885.0,
            total_time_ms=2500.0,
        )
        assert timings.ttft_ms == 150.5
        assert timings.tokens_per_second == 23.4
        assert timings.prompt_tokens_per_second == 885.0
        assert timings.total_time_ms == 2500.0

    def test_timings_optional_fields(self):
        """Test that all Timings fields are optional."""
        timings = Timings()
        assert timings.ttft_ms is None
        assert timings.tokens_per_second is None
        assert timings.prompt_tokens_per_second is None
        assert timings.total_time_ms is None

    def test_timings_partial_fields(self):
        """Test creating Timings with only some fields."""
        timings = Timings(ttft_ms=100.0, tokens_per_second=20.0)
        assert timings.ttft_ms == 100.0
        assert timings.tokens_per_second == 20.0
        assert timings.prompt_tokens_per_second is None
        assert timings.total_time_ms is None

    def test_timings_is_frozen(self):
        """Test that Timings is immutable (frozen)."""
        timings = Timings(ttft_ms=100.0)
        with pytest.raises(Exception):  # ValidationError for frozen model
            timings.ttft_ms = 200.0

    def test_timings_model_dump(self):
        """Test serializing Timings to dict."""
        timings = Timings(ttft_ms=150.0, tokens_per_second=25.0)
        data = timings.model_dump()
        assert data["ttft_ms"] == 150.0
        assert data["tokens_per_second"] == 25.0
        assert data["prompt_tokens_per_second"] is None
        assert data["total_time_ms"] is None


class TestChatCompletionWithTimings:
    """Tests for ChatCompletion with timings field."""

    def test_chat_completion_without_timings(self):
        """Test that ChatCompletion works without timings (backward compatibility)."""
        response = ChatCompletion(
            id="test-id",
            choices=[
                Choice(
                    index=0,
                    message=Message(role="assistant", content="Hello!"),
                    finish_reason="stop",
                )
            ],
            model="test-model",
            provider="test-provider",
        )
        assert response.timings is None
        assert response.content == "Hello!"

    def test_chat_completion_with_timings(self):
        """Test ChatCompletion with timings populated."""
        timings = Timings(
            ttft_ms=145.0,
            tokens_per_second=23.5,
            prompt_tokens_per_second=884.9,
        )
        usage = Usage(prompt_tokens=88, completion_tokens=42, total_tokens=130)

        response = ChatCompletion(
            id="test-id",
            choices=[
                Choice(
                    index=0,
                    message=Message(role="assistant", content="Hello!"),
                    finish_reason="stop",
                )
            ],
            model="llama-3.2-3b",
            provider="llamacpp",
            usage=usage,
            timings=timings,
        )

        assert response.timings is not None
        assert response.timings.ttft_ms == 145.0
        assert response.timings.tokens_per_second == 23.5
        assert response.usage.prompt_tokens == 88

    def test_chat_completion_timings_serialization(self):
        """Test that timings serialize correctly."""
        timings = Timings(ttft_ms=100.0, tokens_per_second=20.0)
        response = ChatCompletion(
            id="test-id",
            choices=[
                Choice(
                    index=0,
                    message=Message(role="assistant", content="Test"),
                    finish_reason="stop",
                )
            ],
            model="test-model",
            provider="test-provider",
            timings=timings,
        )

        data = response.model_dump()
        assert data["timings"]["ttft_ms"] == 100.0
        assert data["timings"]["tokens_per_second"] == 20.0


class TestMetricsLogger:
    """Tests for the MetricsLogger class."""

    def test_logger_creation_default_path(self):
        """Test logger creates with default path."""
        from brio_ext.metrics import MetricsLogger

        logger = MetricsLogger()
        assert logger.log_path.name == "metrics.jsonl"
        assert "briodocs" in str(logger.log_path)

    def test_logger_creation_custom_path(self):
        """Test logger with custom path."""
        from brio_ext.metrics import MetricsLogger

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "custom_metrics.jsonl"
            logger = MetricsLogger(log_path=log_path)
            assert logger.log_path == log_path

    def test_log_basic_record(self):
        """Test logging a basic metrics record."""
        from brio_ext.metrics import MetricsLogger

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test_metrics.jsonl"
            logger = MetricsLogger(log_path=log_path)

            record = logger.log(
                tier_id="fast",
                model="llama-3.2-3b",
                tier_label="Fast",
                ttft_ms=145.0,
                tokens_per_second=23.5,
                prompt_tokens=88,
                completion_tokens=42,
            )

            assert record["tier_id"] == "fast"
            assert record["model"] == "llama-3.2-3b"
            assert record["tier_label"] == "Fast"
            assert record["ttft_ms"] == 145.0
            assert record["tokens_per_second"] == 23.5
            assert "ts" in record

            # Verify file was written
            assert log_path.exists()
            with open(log_path) as f:
                line = f.readline()
                saved_record = json.loads(line)
                assert saved_record["tier_id"] == "fast"

    def test_log_from_response(self):
        """Test logging from response dicts."""
        from brio_ext.metrics import MetricsLogger

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test_metrics.jsonl"
            logger = MetricsLogger(log_path=log_path)

            timings = {"ttft_ms": 150.0, "tokens_per_second": 25.0}
            usage = {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}

            record = logger.log_from_response(
                tier_id="quality",
                model="llama-3.1-8b",
                timings=timings,
                usage=usage,
                tier_label="Quality",
                context_size=8192,
            )

            assert record["tier_id"] == "quality"
            assert record["ttft_ms"] == 150.0
            assert record["prompt_tokens"] == 100
            assert record["context_size"] == 8192

    def test_get_recent(self):
        """Test retrieving recent records."""
        from brio_ext.metrics import MetricsLogger

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test_metrics.jsonl"
            logger = MetricsLogger(log_path=log_path)

            # Log multiple records
            for i in range(5):
                logger.log(tier_id="test", model=f"model-{i}", ttft_ms=100.0 + i)

            records = logger.get_recent(n=3)
            assert len(records) == 3
            # Most recent should be last
            assert records[-1]["model"] == "model-4"
            assert records[0]["model"] == "model-2"

    def test_get_recent_empty_log(self):
        """Test get_recent on empty log."""
        from brio_ext.metrics import MetricsLogger

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test_metrics.jsonl"
            logger = MetricsLogger(log_path=log_path)

            records = logger.get_recent()
            assert records == []

    def test_get_stats(self):
        """Test calculating statistics."""
        from brio_ext.metrics import MetricsLogger

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test_metrics.jsonl"
            logger = MetricsLogger(log_path=log_path)

            # Log records with known values
            logger.log(tier_id="fast", model="test", ttft_ms=100.0, tokens_per_second=20.0)
            logger.log(tier_id="fast", model="test", ttft_ms=200.0, tokens_per_second=30.0)
            logger.log(tier_id="quality", model="test", ttft_ms=300.0, tokens_per_second=15.0)

            # Get stats for all
            stats = logger.get_stats()
            assert stats["count"] == 3
            assert stats["ttft_ms"]["avg"] == 200.0
            assert stats["ttft_ms"]["min"] == 100.0
            assert stats["ttft_ms"]["max"] == 300.0

            # Get stats filtered by tier
            fast_stats = logger.get_stats(tier_id="fast")
            assert fast_stats["count"] == 2
            assert fast_stats["ttft_ms"]["avg"] == 150.0

    def test_clear(self):
        """Test clearing the log."""
        from brio_ext.metrics import MetricsLogger

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test_metrics.jsonl"
            logger = MetricsLogger(log_path=log_path)

            logger.log(tier_id="test", model="test")
            assert log_path.exists()

            logger.clear()
            assert not log_path.exists()

    def test_rounding(self):
        """Test that float values are rounded appropriately."""
        from brio_ext.metrics import MetricsLogger

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test_metrics.jsonl"
            logger = MetricsLogger(log_path=log_path)

            record = logger.log(
                tier_id="test",
                model="test",
                ttft_ms=145.12345,
                tokens_per_second=23.56789,
            )

            assert record["ttft_ms"] == 145.12
            assert record["tokens_per_second"] == 23.57
