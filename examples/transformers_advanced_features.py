"""Example demonstrating advanced Transformers embedding features in Esperanto.

This example shows how to use task optimization, late chunking, and output dimension
control with the Transformers provider for privacy-first advanced embedding capabilities.
"""

from esperanto import AIFactory
from esperanto.common_types.task_type import EmbeddingTaskType


def main():
    """Demonstrate advanced Transformers embedding features."""
    print("🚀 Esperanto Transformers Advanced Features Demo\n")
    
    # Example 1: Task-Specific Optimization
    print("1️⃣  Task-Specific Optimization")
    print("=" * 50)
    
    # Create model with task optimization for retrieval queries
    retrieval_model = AIFactory.create_embedding(
        provider="transformers",
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        config={
            "device": "cpu",  # Use CPU for demo
            "task_type": EmbeddingTaskType.RETRIEVAL_QUERY,
        }
    )
    
    query = "What is machine learning?"
    print(f"Query: {query}")
    embeddings = retrieval_model.embed([query])
    print(f"✅ Generated embedding with task optimization: {len(embeddings[0])} dimensions")
    print()
    
    # Example 2: Late Chunking for Long Documents  
    print("2️⃣  Late Chunking for Long Documents")
    print("=" * 50)
    
    chunking_model = AIFactory.create_embedding(
        provider="transformers",
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        config={
            "device": "cpu",
            "task_type": EmbeddingTaskType.RETRIEVAL_DOCUMENT,
            "late_chunking": True,
        }
    )
    
    long_document = """
    Artificial intelligence (AI) refers to the simulation of human intelligence in machines that are programmed to think and act like humans. The term may also be applied to any machine that exhibits traits associated with a human mind such as learning and problem-solving. The ideal characteristic of artificial intelligence is its ability to rationalize and take actions that have the best chance of achieving a specific goal. Machine learning is a subset of artificial intelligence that refers to the automatic improvement of computer programs through experience. Deep learning is a subset of machine learning that uses neural networks with multiple layers to model and understand complex patterns in data. Natural language processing is another important area of AI that focuses on the interaction between computers and humans through natural language.
    """
    
    print(f"Document length: {len(long_document)} characters")
    embeddings = chunking_model.embed([long_document])
    print(f"✅ Generated embedding with late chunking: {len(embeddings[0])} dimensions")
    print()
    
    # Example 3: Output Dimension Control
    print("3️⃣  Output Dimension Control")
    print("=" * 50)
    
    # Dimension reduction
    compact_model = AIFactory.create_embedding(
        provider="transformers",
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        config={
            "device": "cpu",
            "output_dimensions": 128,  # Reduce from 384 to 128
        }
    )
    
    text = "This text will be embedded in lower dimensions"
    print(f"Text: {text}")
    compact_embeddings = compact_model.embed([text])
    print(f"✅ Generated compact embedding: {len(compact_embeddings[0])} dimensions (reduced from 384)")
    
    # Dimension expansion
    expanded_model = AIFactory.create_embedding(
        provider="transformers", 
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        config={
            "device": "cpu",
            "output_dimensions": 512,  # Expand from 384 to 512
        }
    )
    
    expanded_embeddings = expanded_model.embed([text])
    print(f"✅ Generated expanded embedding: {len(expanded_embeddings[0])} dimensions (expanded from 384)")
    print()
    
    # Example 4: All Features Combined
    print("4️⃣  All Advanced Features Combined")
    print("=" * 50)
    
    advanced_model = AIFactory.create_embedding(
        provider="transformers",
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        config={
            "device": "mps",
            "task_type": EmbeddingTaskType.CLASSIFICATION,
            "late_chunking": True,
            "output_dimensions": 256,
            "truncate_at_max_length": True,
        }
    )
    
    texts = [
        "This is a positive review of the product.",
        "I really enjoyed reading this book. It was fantastic and well-written with great character development and an engaging plot that kept me hooked from beginning to end.",
        "The service was terrible and disappointing."
    ]
    
    print("Texts for classification:")
    for i, text in enumerate(texts, 1):
        print(f"  {i}. {text}")
    
    embeddings = advanced_model.embed(texts)
    print(f"\n✅ Generated {len(embeddings)} embeddings with all advanced features:")
    print(f"   - Task optimization: Classification")
    print(f"   - Late chunking: Enabled")
    print(f"   - Output dimensions: {len(embeddings[0])}")
    print()
    
    # Example 5: Qwen3-Embedding-4B with Large Context
    print("5️⃣  Qwen3-Embedding with Large Context (Advanced Model)")
    print("=" * 50)
    
    try:
        qwen_model = AIFactory.create_embedding(
            provider="transformers",
            model_name="Qwen/Qwen3-Embedding-0.6B",
            config={
                "device": "mps",
                "task_type": EmbeddingTaskType.RETRIEVAL_DOCUMENT,
                "late_chunking": True,
                "output_dimensions": 1024,  # Leverage Qwen3's high dimensionality
            }
        )
        
        # Create a very long document that would benefit from Qwen3's large context
        very_long_document = " ".join([
            f"This is sentence {i} in a very long research document about artificial intelligence and machine learning."
            for i in range(200)  # 200 sentences
        ])
        
        print(f"Document length: {len(very_long_document)} characters")
        print("⚠️  Note: Qwen3-Embedding-4B is a large model - this may take some time...")
        
        qwen_embeddings = qwen_model.embed([very_long_document])
        print(f"✅ Generated Qwen3 embedding: {len(qwen_embeddings[0])} dimensions")
        print("✨ Qwen3 handled the large context with advanced chunking!")
        
    except Exception as e:
        print(f"⚠️  Qwen3-Embedding-4B not available or failed to load: {e}")
        print("   This is normal if the model hasn't been downloaded yet.")
    
    print()
    print("🎉 Advanced Features Demo Complete!")
    print("\n💡 Key Benefits:")
    print("   • Privacy-first: All processing happens locally")
    print("   • Cost-effective: No per-embedding API charges")  
    print("   • Advanced capabilities: Task optimization, chunking, dimension control")
    print("   • Universal interface: Same API as cloud providers")
    print("   • Model flexibility: Works with any HuggingFace transformer model")


if __name__ == "__main__":
    # Check if advanced dependencies are available
    try:
        import sentence_transformers
        import sklearn
        print("✅ Advanced dependencies available - all features enabled")
        main()
    except ImportError as e:
        print("⚠️  Advanced dependencies not available:")
        print(f"   Missing: {e}")
        print("\n💻 To enable advanced features, install with:")
        print("   pip install esperanto[transformers]")
        print("\n🔧 This will install:")
        print("   • sentence-transformers (for semantic chunking)")
        print("   • scikit-learn (for PCA dimension reduction)")
        print("   • Additional ML dependencies")