from sentence_transformers import SentenceTransformer
import numpy as np

# Load efficient embedding model
# Ensure this model name is correct and accessible
# Using a try-except block for robustness in case model loading fails
try:
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
except Exception as e:
    print(f"Error loading SentenceTransformer model: {e}")
    print("Please ensure 'all-MiniLM-L6-v2' is installed or accessible.")
    # Fallback to a dummy embedder if loading fails, to allow basic system operation
    class DummyEmbedder:
        def encode(self, texts):
            print(f"DummyEmbedder: Simulating embedding for: {texts}")
            # Return a zero vector of the expected dimension
            # This is a placeholder and won't produce meaningful results
            return np.zeros((len(texts), 384), dtype=np.float32)
    embedder = DummyEmbedder()


def get_embedding(text: str) -> np.ndarray:
    # The model expects a list of texts and returns a list of embeddings.
    # We are embedding a single text, so we pass [text] and take the first result [0].
    embedding = embedder.encode([text])[0]
    return embedding.astype(np.float32) # Ensure it's float32 for FAISS
