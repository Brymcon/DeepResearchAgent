import numpy as np
from memory.store import VectorMemory # This will be the new DuckDB version
from memory.decay import calculate_decayed_importance
import time

class MemoryAgent:
    def __init__(self):
        # Initialize VectorMemory (DuckDB-based)
        # The VectorMemory __init__ handles DB connection and table setup.
        try:
            self.memory = VectorMemory()
            print("MemoryAgent: Successfully initialized DuckDB-based VectorMemory.")
        except Exception as e:
            print(f"MemoryAgent: CRITICAL ERROR initializing VectorMemory: {e}")
            # In a real application, might want to raise this or enter a degraded state
            self.memory = None # Indicate failure

    def retrieve(self, query_embedding: np.ndarray, k: int = 5, threshold: float = None):
        if not self.memory:
            print("MemoryAgent: Memory system not available.")
            return []
        try:
            return self.memory.retrieve(query_embedding, k, threshold)
        except Exception as e:
            print(f"MemoryAgent: Error during memory retrieval: {e}")
            return []

    def store(self, embedding: np.ndarray, content: str, tags: list[str] = None,
              source: str = "unknown", certainty: float = 1.0,
              initial_decay_rate: float = 0.95, memory_type: str = "fact"):
        if not self.memory:
            print("MemoryAgent: Memory system not available.")
            return None
        try:
            return self.memory.add_memory(embedding, content, tags, source, certainty, initial_decay_rate, memory_type)
        except Exception as e:
            print(f"MemoryAgent: Error during memory storage: {e}")
            return None

    def get_memory_by_id(self, memory_id: int):
        if not self.memory:
            print("MemoryAgent: Memory system not available.")
            return None
        try:
            return self.memory.get_memory_by_id(memory_id)
        except Exception as e:
            print(f"MemoryAgent: Error retrieving memory by ID {memory_id}: {e}")
            return None

    def maintain(self, max_age_days: int = 30, relevance_boost_factor: float = 0.05, importance_floor: float = 0.1):
        if not self.memory:
            print("MemoryAgent: Memory system not available. Cannot perform maintenance.")
            return

        print("MemoryAgent: Starting memory maintenance cycle...")
        try:
            memories_for_decay = self.memory.get_all_memories_for_decay()
            if not memories_for_decay:
                print("MemoryAgent: No memories found for decay processing.")
                return

            current_time = time.time()
            updated_count = 0
            deleted_count = 0

            for mem_tuple in memories_for_decay:
                # Convert tuple from DB to dict for calculate_decayed_importance
                # Expected order: id, timestamp, importance, access_count, initial_decay_rate, last_accessed_ts
                memory_data = {
                    'id': mem_tuple[0],
                    'timestamp': mem_tuple[1],
                    'importance': mem_tuple[2],
                    'access_count': mem_tuple[3],
                    'initial_decay_rate': mem_tuple[4],
                    'last_accessed_ts': mem_tuple[5]
                }

                new_importance, should_delete = calculate_decayed_importance(
                    memory_data, current_time, max_age_days, relevance_boost_factor, importance_floor
                )

                if should_delete:
                    self.memory.delete_memory(memory_data['id'])
                    deleted_count += 1
                elif abs(new_importance - memory_data['importance']) > 1e-5: # Update only if changed significantly
                    self.memory.update_memory_importance(memory_data['id'], new_importance)
                    updated_count += 1

            # Explicitly save (checkpoint) DuckDB after maintenance
            self.memory.save()
            print(f"MemoryAgent: Maintenance complete. Updated: {updated_count}, Deleted: {deleted_count}.")

        except Exception as e:
            print(f"MemoryAgent: Error during memory maintenance: {e}")

    def close_memory(self):
        if self.memory:
            print("MemoryAgent: Closing memory connection.")
            self.memory.close()
