"""Example usage of Jina embeddings with task-aware features."""

import os
from esperanto import AIFactory
from esperanto.common_types.task_type import EmbeddingTaskType


def basic_example():
    """Basic usage of Jina embeddings."""
    print("=== Basic Jina Embeddings ===")
    
    # Create a basic Jina embedding model
    model = AIFactory.create_embedding("jina", "jina-embeddings-v3")
    
    # Generate embeddings
    texts = ["Hello, world!", "How are you today?"]
    embeddings = model.embed(texts)
    
    print(f"Number of texts: {len(texts)}")
    print(f"Number of embeddings: {len(embeddings)}")
    print(f"Embedding dimension: {len(embeddings[0])}")
    print(f"First few values: {embeddings[0][:5]}")
    print()


def task_aware_example():
    """Example using task-aware embeddings."""
    print("=== Task-Aware Embeddings ===")
    
    # Create models optimized for different tasks
    query_model = AIFactory.create_embedding(
        provider="jina",
        model_name="jina-embeddings-v3",
        config={
            "task_type": EmbeddingTaskType.RETRIEVAL_QUERY,
            "output_dimensions": 512
        }
    )
    
    doc_model = AIFactory.create_embedding(
        provider="jina",
        model_name="jina-embeddings-v3",
        config={
            "task_type": EmbeddingTaskType.RETRIEVAL_DOCUMENT,
            "output_dimensions": 512
        }
    )
    
    # Example query and documents
    query = "What is machine learning?"
    documents = [
        "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
        "The weather today is sunny and warm.",
        "Deep learning uses neural networks with multiple layers to process complex patterns."
    ]
    
    # Generate embeddings
    query_embedding = query_model.embed([query])[0]
    doc_embeddings = doc_model.embed(documents)
    
    print(f"Query embedding dimension: {len(query_embedding)}")
    print(f"Document embeddings: {len(doc_embeddings)} documents")
    
    # Calculate similarity (cosine similarity)
    import numpy as np
    
    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    print("\nSimilarity scores:")
    for i, doc in enumerate(documents):
        similarity = cosine_similarity(query_embedding, doc_embeddings[i])
        print(f"  Document {i+1}: {similarity:.4f} - {doc[:50]}...")
    print()


def advanced_features_example():
    """Example using advanced features like late chunking."""
    print("=== Advanced Features ===")
    
    # Create model with late chunking enabled
    model = AIFactory.create_embedding(
        provider="jina",
        model_name="jina-embeddings-v3",
        config={
            "task_type": EmbeddingTaskType.RETRIEVAL_DOCUMENT,
            "late_chunking": True,
            "output_dimensions": 1024
        }
    )
    
    # Long document that benefits from late chunking
    long_doc = """
    Artificial intelligence (AI) is transforming how we live and work. 
    From natural language processing that powers chatbots to computer vision 
    systems that can identify objects in images, AI is becoming increasingly 
    sophisticated. Machine learning algorithms learn patterns from data, 
    while deep learning uses neural networks to tackle complex problems. 
    The future of AI promises even more breakthroughs in areas like 
    autonomous vehicles, medical diagnosis, and scientific research.
    """
    
    embedding = model.embed([long_doc])[0]
    print(f"Embedding dimension with late chunking: {len(embedding)}")
    print()


def universal_interface_example():
    """Demonstrate universal interface working across providers."""
    print("=== Universal Interface ===")
    
    # Same config works for any provider!
    config = {
        "task_type": EmbeddingTaskType.CLASSIFICATION,
        "output_dimensions": 256
    }
    
    # This would work with any provider that supports embeddings
    # For demo, we'll just show Jina
    providers = ["jina"]  # Could add: "openai", "gemini", "transformers"
    
    for provider in providers:
        try:
            model = AIFactory.create_embedding(
                provider=provider,
                model_name=None,  # Use default
                config=config
            )
            
            result = model.embed(["Sample text for classification"])
            print(f"{provider}: Generated {len(result[0])}-dimensional embedding")
        except Exception as e:
            print(f"{provider}: {e}")
    print()


if __name__ == "__main__":
    # Check if API key is set
    if not os.getenv("JINA_API_KEY"):
        print("Please set JINA_API_KEY environment variable")
        print("You can get one at https://jina.ai/")
        exit(1)
    
    # Run examples
    basic_example()
    task_aware_example()
    advanced_features_example()
    universal_interface_example()
    
    print("✅ All examples completed successfully!")