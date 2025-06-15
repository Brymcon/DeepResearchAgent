from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Optional

# Global cache for sentence transformer models
_model_cache = {}

DEFAULT_MODEL_NAME = 'all-MiniLM-L6-v2' # 384 dimensions
DEFAULT_DIMENSIONS = 384

def get_sentence_embedding_model(model_name: str = DEFAULT_MODEL_NAME) -> Optional[SentenceTransformer]:
    """Loads a sentence-transformer model, caching it for future use."""
    if model_name in _model_cache:
        return _model_cache[model_name]
    try:
        print(f"EmbeddingUtils: Loading sentence transformer model '{model_name}'...")
        model = SentenceTransformer(model_name)
        _model_cache[model_name] = model
        print(f"EmbeddingUtils: Model '{model_name}' loaded and cached.")
        return model
    except Exception as e:
        print(f"EmbeddingUtils: CRITICAL ERROR - Failed to load sentence transformer model '{model_name}': {e}")
        print("EmbeddingUtils: Falling back to no model. Embeddings will be zero vectors.")
        _model_cache[model_name] = None # Cache the failure to avoid retrying
        return None

def get_sentence_embedding(text: str, model_name: str = DEFAULT_MODEL_NAME, target_dim: int = DEFAULT_DIMENSIONS) -> List[float]:
    """
    Generates a sentence embedding for the given text using a specified sentence-transformer model.
    Args:
        text: The input text to embed.
        model_name: The name of the sentence-transformer model to use.
        target_dim: The expected dimension of the output embedding. Used for zero vector fallback.
    Returns:
        A list of floats representing the embedding, or a list of zeros if embedding fails.
    """
    if not text or not isinstance(text, str):
        # print(f"EmbeddingUtils: Warning - Invalid input text for embedding. Returning zero vector.")
        return [0.0] * target_dim

    model = get_sentence_embedding_model(model_name)

    if model is None:
        return [0.0] * target_dim

    try:
        embedding_array = model.encode([text], convert_to_numpy=True)[0]
        # Ensure the output is a flat list of Python floats
        return embedding_array.astype(float).tolist()
    except Exception as e:
        print(f"EmbeddingUtils: Error generating sentence embedding for text '{text[:50]}...': {e}")
        return [0.0] * target_dim

# Example usage:
# if __name__ == '__main__':
#     test_texts = [
#         "This is a test sentence.",
#         "Hello world!",
#         ""
#     ]
#     for text in test_texts:
#         embedding = get_sentence_embedding(text)
#         print(f"Text: '{text}'")
#         if embedding:
#             print(f"  Embedding (first 5 dims): {embedding[:5]}")
#             print(f"  Embedding dimension: {len(embedding)}")
#         else:
#             print("  Failed to generate embedding.")

#     # Test with a different model (if available and you want to test loading another)
#     # embedding_mpnet = get_sentence_embedding("Test sentence with another model", model_name='all-mpnet-base-v2', target_dim=768)
#     # if embedding_mpnet:
#     #     print(f"MPNet Embedding dim: {len(embedding_mpnet)}")
