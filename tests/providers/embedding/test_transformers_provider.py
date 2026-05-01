"""Tests for the Transformers embedding model provider."""

import importlib.metadata
import sys
from unittest.mock import MagicMock, patch

import pytest

from esperanto.factory import AIFactory


@pytest.fixture
def model():
    """Create a test model instance."""
    return AIFactory.create_embedding(
        provider="transformers",
        model_name="bert-base-uncased",
        config={
            "device": "cpu",  # Force CPU for testing
            "pooling_strategy": "mean",
        },
    )


def test_initialization(model):
    """Test model initialization."""
    assert model.provider == "transformers"
    assert model.get_model_name() == "bert-base-uncased"
    assert model.device == "cpu"
    assert model.pooling_config.strategy == "mean"
    assert model.pooling_config.attention_mask is True


def test_embed(model):
    """Test embedding generation."""
    texts = ["Hello world", "This is a test"]
    embeddings = model.embed(texts)

    assert len(embeddings) == len(texts)
    assert all(isinstance(emb, list) for emb in embeddings)
    assert all(isinstance(val, float) for emb in embeddings for val in emb)
    # BERT base has 768-dimensional embeddings
    assert all(len(emb) == 768 for emb in embeddings)


@pytest.mark.asyncio
async def test_aembed(model):
    """Test async embedding generation."""
    texts = ["Hello world", "This is a test"]
    embeddings = await model.aembed(texts)

    assert len(embeddings) == len(texts)
    assert all(isinstance(emb, list) for emb in embeddings)
    assert all(isinstance(val, float) for emb in embeddings for val in emb)
    # BERT base has 768-dimensional embeddings
    assert all(len(emb) == 768 for emb in embeddings)


def test_pooling_strategies(model):
    """Test different pooling strategies."""
    text = ["Test text"]

    # Test mean pooling (default)
    mean_embeddings = model.embed(text)

    # Test max pooling
    model.pooling_config.strategy = "max"
    max_embeddings = model.embed(text)

    # Test cls pooling
    model.pooling_config.strategy = "cls"
    cls_embeddings = model.embed(text)

    # All should have same dimensions but different values
    assert len(mean_embeddings[0]) == len(max_embeddings[0]) == len(cls_embeddings[0])
    assert mean_embeddings != max_embeddings
    assert mean_embeddings != cls_embeddings
    assert max_embeddings != cls_embeddings


def test_invalid_texts(model):
    """Test error handling for invalid inputs."""
    with pytest.raises(ValueError, match="Texts cannot be empty"):
        model.embed([])


def test_model_list():
    """Test listing available models."""
    models = AIFactory.get_available_providers()
    assert "transformers" in models["embedding"]


def test_quantization():
    """Test model initialization with quantization."""
    pytest.importorskip("bitsandbytes")

    model = AIFactory.create_embedding(
        provider="transformers",
        model_name="bert-base-uncased",
        config={
            "device": "cpu",
            "quantize": "8bit",
        },
    )

    # Basic functionality test with quantized model
    embeddings = model.embed(["Test text"])
    assert len(embeddings) == 1
    assert len(embeddings[0]) == 768


def _mock_metadata_version(name: str) -> str:
    if name == "bitsandbytes":
        return "0.42.0"
    return importlib.metadata.version(name)


def test_quantization_4bit_uses_bits_and_bytes_config():
    """Test that 4bit quantization passes a BitsAndBytesConfig to from_pretrained."""
    from transformers import BitsAndBytesConfig

    from esperanto.providers.embedding.transformers import TransformersEmbeddingModel

    mock_bnb = MagicMock()
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    bnb_modules = {
        "bitsandbytes": mock_bnb,
        "bitsandbytes.nn": mock_bnb.nn,
        "bitsandbytes.functional": mock_bnb.functional,
    }

    with (
        patch.dict(sys.modules, bnb_modules),
        patch("importlib.metadata.version", side_effect=_mock_metadata_version),
        patch(
            "esperanto.providers.embedding.transformers.AutoModel.from_pretrained",
            return_value=mock_model,
        ) as mock_from_pretrained,
        patch(
            "esperanto.providers.embedding.transformers.AutoTokenizer.from_pretrained",
            return_value=mock_tokenizer,
        ),
    ):
        TransformersEmbeddingModel(
            model_name="bert-base-uncased",
            device="cpu",
            quantize="4bit",
        )

        call_kwargs = mock_from_pretrained.call_args.kwargs
        assert "quantization_config" in call_kwargs
        bnb_config = call_kwargs["quantization_config"]
        assert isinstance(bnb_config, BitsAndBytesConfig)
        assert bnb_config.load_in_4bit is True
        assert bnb_config.load_in_8bit is False


def test_quantization_8bit_uses_bits_and_bytes_config():
    """Test that 8bit quantization passes a BitsAndBytesConfig to from_pretrained."""
    from transformers import BitsAndBytesConfig

    from esperanto.providers.embedding.transformers import TransformersEmbeddingModel

    mock_bnb = MagicMock()
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    bnb_modules = {
        "bitsandbytes": mock_bnb,
        "bitsandbytes.nn": mock_bnb.nn,
        "bitsandbytes.functional": mock_bnb.functional,
    }

    with (
        patch.dict(sys.modules, bnb_modules),
        patch(
            "esperanto.providers.embedding.transformers.AutoModel.from_pretrained",
            return_value=mock_model,
        ) as mock_from_pretrained,
        patch(
            "esperanto.providers.embedding.transformers.AutoTokenizer.from_pretrained",
            return_value=mock_tokenizer,
        ),
    ):
        TransformersEmbeddingModel(
            model_name="bert-base-uncased",
            device="cpu",
            quantize="8bit",
        )

        call_kwargs = mock_from_pretrained.call_args.kwargs
        assert "quantization_config" in call_kwargs
        bnb_config = call_kwargs["quantization_config"]
        assert isinstance(bnb_config, BitsAndBytesConfig)
        assert bnb_config.load_in_8bit is True
        assert bnb_config.load_in_4bit is False
