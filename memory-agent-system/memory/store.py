import faiss
import numpy as np
import json
import os
import time # Added import for time

class VectorMemory:
    def __init__(self, dim=384):  # Dimension for all-MiniLM-L6-v2
        self.dim = dim
        self.index = faiss.IndexFlatL2(dim)
        self.metadata = []
        self.counter = 0

    def add_memory(self, embedding: np.ndarray, content: str, memory_type: str = "fact"):
        # Ensure proper shape
        embedding = embedding.reshape(1, -1).astype('float32')
        self.index.add(embedding)

        # Store metadata
        self.metadata.append({
            "id": self.counter,
            "content": content,
            "type": memory_type,
            "timestamp": time.time(),
            "access_count": 0
        })
        self.counter += 1
        return self.counter - 1

    def retrieve(self, query_embedding: np.ndarray, k=5):
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        distances, indices = self.index.search(query_embedding, k)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx >= 0:  # FAISS returns -1 for invalid indices
                memory = self.metadata[idx]
                memory["access_count"] += 1
                results.append({
                    "id": memory["id"],
                    "content": memory["content"],
                    "distance": float(distances[0][i]),
                    "type": memory["type"]
                })
        return results

    def save(self, path="memory_data"):
        os.makedirs(path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(path, "vector.index"))
        with open(os.path.join(path, "metadata.json"), 'w') as f:
            json.dump(self.metadata, f)

    def load(self, path="memory_data"):
        self.index = faiss.read_index(os.path.join(path, "vector.index"))
        with open(os.path.join(path, "metadata.json"), 'r') as f:
            self.metadata = json.load(f)
        self.counter = max([m['id'] for m in self.metadata]) + 1 if self.metadata else 0
