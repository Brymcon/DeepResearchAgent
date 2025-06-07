import numpy as np # Added import for np.ndarray
from memory.store import VectorMemory
from memory.decay import apply_memory_decay

class MemoryAgent:
    def __init__(self):
        self.memory = VectorMemory()
        try:
            self.memory.load()
        except:
            print("Starting with fresh memory")

    def retrieve(self, query_embedding: np.ndarray, k=5): # Added np.ndarray type hint
        return self.memory.retrieve(query_embedding, k)

    def store(self, embedding: np.ndarray, content: str, memory_type: str): # Added np.ndarray type hint
        return self.memory.add_memory(embedding, content, memory_type)

    def maintain(self):
        self.memory.metadata = apply_memory_decay(self.memory.metadata)
        self.memory.save()
