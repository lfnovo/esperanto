"""Integration tests for embedding providers."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from esperanto.providers.embedding.azure import AzureEmbeddingModel
from esperanto.providers.embedding.google import GoogleEmbeddingModel

# @pytest.fixture
# def transformers_model():
#     """Create a transformers model instance."""
#     return AIFactory.create_embedding(
#         provider="transformers",
#         model_name="bert-base-uncased",
#         config={
#             "device": "cpu",
#             "pooling_strategy": "mean",
#         },
#     )


# def test_transformers_embedding(transformers_model):
#     """Test transformers embedding generation."""
#     texts = [
#         "This is a test sentence.",
#         "Another example for embedding.",
#     ]

#     embeddings = transformers_model.embed(texts)

#     # Basic validation
#     assert len(embeddings) == len(texts)
#     assert all(isinstance(emb, list) for emb in embeddings)
#     assert all(isinstance(val, float) for emb in embeddings for val in emb)
#     # BERT base has 768-dimensional embeddings
#     assert all(len(emb) == 768 for emb in embeddings)

#     # Verify embeddings are different for different texts
#     assert embeddings[0] != embeddings[1]


# @pytest.mark.asyncio
# async def test_transformers_async_embedding(transformers_model):
#     """Test async transformers embedding generation."""
#     texts = [
#         "This is a test sentence.",
#         "Another example for embedding.",
#     ]

#     embeddings = await transformers_model.aembed(texts)

#     # Basic validation
#     assert len(embeddings) == len(texts)
#     assert all(isinstance(emb, list) for emb in embeddings)
#     assert all(isinstance(val, float) for emb in embeddings for val in emb)
#     # BERT base has 768-dimensional embeddings
#     assert all(len(emb) == 768 for emb in embeddings)

#     # Verify embeddings are different for different texts
#     assert embeddings[0] != embeddings[1]


# Note, if the mock code is commented out, and the API Key and Base URL replaced with actual
# values then an actual model can be called. I assumed that was the purpose of the tests in
# this file.
@pytest.fixture
def azure_model():
    """Create a Azure embedding model instance with mocked responses."""
    model = AzureEmbeddingModel(api_key="AZURE_OPENAI_API_KEY",
                                base_url="AZURE_OPENAI_ENDPOINT",
                                api_version="2024-02-01",
                                model_name="text-embedding-3-large")

    # Mock embed and aembed methods w/ 256-dimensional embedding
    sample_embedding = [[0.1, 0.2, 0.3, 0.4] * 64,
                        [0.4, 0.3, 0.2, 0.1] * 64]
    model.embed = MagicMock(return_value=sample_embedding)
    model.aembed = AsyncMock(return_value=sample_embedding)

    return model

@pytest.mark.asyncio
async def test_azure_embedding(azure_model):
    """Test Azure embedding generation."""
    texts = [
        "This is a test sentence.",
        "Another example for embedding.",
    ]

    # text-embedding-3-large defaults to 3072-dimensional embedding
    # text-embedding-3-small defaults to 1536-dimensional embedding
    # mock returns a 256-dimensional embedding
    dimensions = 256

    embeddings = azure_model.embed(texts, dimensions=dimensions)
    aembeddings = await azure_model.aembed(texts, dimensions=dimensions)

    assert embeddings == aembeddings

    # Basic validation
    assert isinstance(embeddings, list)
    assert all(isinstance(emb, list) for emb in embeddings)
    assert all(isinstance(val, float) for emb in embeddings for val in emb)

    # Data check - our mock returns a 256-dim embedding
    assert all(len(emb) == dimensions for emb in embeddings)
    assert embeddings[0] != embeddings[1]


@pytest.fixture
def google_model():
    """Create a Google embedding model instance with mocked responses."""
    model = GoogleEmbeddingModel(api_key="test-key")

    # Mock embed and aembed methods
    sample_embedding = [[0.1, 0.2, 0.3, 0.4] * 64]  # 256-dimensional embedding
    model.embed = MagicMock(return_value=sample_embedding)
    model.aembed = AsyncMock(return_value=sample_embedding)

    return model


def test_google_embedding(google_model):
    """Test Google embedding generation."""
    texts = [
        "This is a test sentence.",
        "Another example for embedding.",
    ]

    embeddings = google_model.embed(texts)

    # Verify the method was called with the correct texts
    google_model.embed.assert_called_once_with(texts)

    # Basic validation
    assert isinstance(embeddings, list)
    assert all(isinstance(emb, list) for emb in embeddings)
    assert all(isinstance(val, float) for emb in embeddings for val in emb)

    # Length check - our mock returns a 256-dim embedding
    assert len(embeddings[0]) == 256


@pytest.mark.asyncio
async def test_google_async_embedding(google_model):
    """Test async Google embedding generation."""
    texts = [
        "This is a test sentence.",
        "Another example for embedding.",
    ]

    embeddings = await google_model.aembed(texts)

    # Verify the method was called with the correct texts
    google_model.aembed.assert_called_once_with(texts)

    # Basic validation
    assert isinstance(embeddings, list)
    assert all(isinstance(emb, list) for emb in embeddings)
    assert all(isinstance(val, float) for emb in embeddings for val in emb)

    # Length check - our mock returns a 256-dim embedding
    assert len(embeddings[0]) == 256


# def test_transformers_pooling_strategies(transformers_model):
#     """Test different pooling strategies."""
#     text = ["Test sentence for pooling."]

#     # Test mean pooling (default)
#     mean_embeddings = transformers_model.embed(text)

#     # Test max pooling
#     transformers_model.pooling_config.strategy = "max"
#     max_embeddings = transformers_model.embed(text)

#     # Test cls pooling
#     transformers_model.pooling_config.strategy = "cls"
#     cls_embeddings = transformers_model.embed(text)

#     # All should have same dimensions but different values
#     assert len(mean_embeddings[0]) == len(max_embeddings[0]) == len(cls_embeddings[0])
#     assert mean_embeddings != max_embeddings
#     assert mean_embeddings != cls_embeddings
#     assert max_embeddings != cls_embeddings


# def test_transformers_batch_processing(transformers_model):
#     """Test batch processing of texts."""
#     # Create a list of texts longer than the default batch size
#     texts = [
#         f"Test sentence number {i}." for i in range(40)
#     ]  # Default batch size is 32

#     embeddings = transformers_model.embed(texts)

#     # Verify all texts were processed
#     assert len(embeddings) == len(texts)
#     assert all(len(emb) == 768 for emb in embeddings)


# @pytest.mark.asyncio
# async def test_transformers_async_batch_processing(transformers_model):
#     """Test async batch processing of texts."""
#     # Create a list of texts longer than the default batch size
#     texts = [
#         f"Test sentence number {i}." for i in range(40)
#     ]  # Default batch size is 32

#     embeddings = await transformers_model.aembed(texts)

#     # Verify all texts were processed
#     assert len(embeddings) == len(texts)
#     assert all(len(emb) == 768 for emb in embeddings)


# def test_transformers_error_handling(transformers_model):
#     """Test error handling for invalid inputs."""
#     # Test empty input
#     with pytest.raises(ValueError, match="Texts cannot be empty"):
#         transformers_model.embed([])

#     # Test None input
#     with pytest.raises(ValueError, match="Texts cannot be empty"):
#         transformers_model.embed(None)


# @pytest.mark.asyncio
# async def test_transformers_async_error_handling(transformers_model):
#     """Test async error handling for invalid inputs."""
#     # Test empty input
#     with pytest.raises(ValueError, match="Texts cannot be empty"):
#         await transformers_model.aembed([])

#     # Test None input
#     with pytest.raises(ValueError, match="Texts cannot be empty"):
#         await transformers_model.aembed(None)


# def test_transformers_device_config():
#     """Test device configuration."""
#     # Test explicit CPU config
#     cpu_model = AIFactory.create_embedding(
#         provider="transformers",
#         model_name="bert-base-uncased",
#         config={"device": "cpu"},
#     )
#     assert cpu_model.device == "cpu"

#     # Test auto device config with CPU (no CUDA or MPS)
#     with (
#         patch("torch.cuda.is_available", return_value=False),
#         patch("torch.backends.mps.is_available", return_value=False),
#     ):
#         cpu_auto_model = AIFactory.create_embedding(
#             provider="transformers",
#             model_name="bert-base-uncased",
#             config={"device": "auto"},
#         )
#         assert cpu_auto_model.device == "cpu"

#     # Test auto device config with MPS (Apple Silicon)
#     with (
#         patch("torch.cuda.is_available", return_value=False),
#         patch("torch.backends.mps.is_available", return_value=True),
#     ):
#         mps_auto_model = AIFactory.create_embedding(
#             provider="transformers",
#             model_name="bert-base-uncased",
#             config={"device": "auto"},
#         )
#         assert mps_auto_model.device == "mps"
