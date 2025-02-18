"""Test script for transformers embedding provider with different pooling strategies."""

import numpy as np

from esperanto.factory import AIFactory


def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def test_pooling_strategies():
    """Test different pooling strategies and compare embeddings."""
    # Test texts with various relationships
    texts = [
        "O gato pulou o muro.",  # Base sentence in Portuguese
        "The cat jumped over the wall.",  # Same meaning in English
        "Um gato saltou a cerca.",  # Similar meaning, different words
        "O cachorro late muito alto.",  # Different meaning, same language
        "The dog is sleeping quietly.",  # Different meaning, different language
        "O muro é muito alto para pular.",  # Related context, different focus
    ]

    # Create model and test different pooling strategies
    model = AIFactory.create_embedding(
        provider="transformers",
        model_name="bert-base-multilingual-cased",  # Good for multiple languages
        config={
            "device": "cpu",
            "pooling_strategy": "mean",
        },
    )

    # Get embeddings with mean pooling
    print("\nGerando embeddings com diferentes estratégias de pooling...")
    print("\nMean Pooling:")
    mean_embeddings = model.embed(texts)

    # Switch to max pooling
    print("\nMax Pooling:")
    model.pooling_config.strategy = "max"
    max_embeddings = model.embed(texts)

    # Switch to cls pooling
    print("\nCLS Pooling:")
    model.pooling_config.strategy = "cls"
    cls_embeddings = model.embed(texts)

    def print_similarities(name, embeddings):
        """Print similarities for a given pooling strategy."""
        print(f"\n{name} Pooling:")
        print("Comparando com a frase base 'O gato pulou o muro':")
        similarities = []
        for i, text in enumerate(texts[1:], 1):
            similarity = cosine_similarity(embeddings[0], embeddings[i])
            similarities.append((text, similarity))
            print(f"- vs '{text}': {similarity:.4f}")

        # Find most and least similar
        most_similar = max(similarities, key=lambda x: x[1])
        least_similar = min(similarities, key=lambda x: x[1])
        print(f"\nMais similar: '{most_similar[0]}' ({most_similar[1]:.4f})")
        print(f"Menos similar: '{least_similar[0]}' ({least_similar[1]:.4f})")

    # Compare similarities for each pooling strategy
    print_similarities("Mean", mean_embeddings)
    print_similarities("Max", max_embeddings)
    print_similarities("CLS", cls_embeddings)

    # Print embedding dimensions and analysis
    print(f"\nDimensões dos embeddings: {len(mean_embeddings[0])}")
    print("\nAnálise das estratégias de pooling:")
    print("- Mean Pooling: Mais discriminativa, melhor para comparações semânticas")
    print(
        "- Max Pooling: Similaridades mais altas em geral, boa para features importantes"
    )
    print(
        "- CLS Pooling: Similaridades muito altas, melhor para representação geral da frase"
    )


if __name__ == "__main__":
    test_pooling_strategies()
