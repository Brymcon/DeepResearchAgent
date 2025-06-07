import numpy as np
from memory.store import VectorMemory
from memory.decay import calculate_decayed_importance
import time
import datetime # For timestamp math if needed for transitions

class MemoryAgent:
    def __init__(self):
        try:
            self.memory = VectorMemory()
            print("MemoryAgent: Successfully initialized DuckDB-based VectorMemory.")
        except Exception as e:
            print(f"MemoryAgent: CRITICAL ERROR initializing VectorMemory: {e}")
            self.memory = None

    def retrieve(self, query_embedding: np.ndarray, k: int = 5, threshold: float = None, bucket_types: list[str] = None):
        if not self.memory:
            print("MemoryAgent: Memory system not available.")
            return []
        try:
            # Default to searching short and mid term if no specific buckets provided
            if bucket_types is None:
                bucket_types = ['short_term', 'mid_term']
            return self.memory.retrieve(query_embedding, k, threshold, bucket_types)
        except Exception as e:
            print(f"MemoryAgent: Error during memory retrieval: {e}")
            return []

    def store(self, embedding: np.ndarray, content: str, tags: list[str] = None,
              source: str = "unknown", certainty: float = 1.0,
              initial_decay_rate: float = 0.95, memory_type: str = "fact",
              bucket_type: str = 'short_term'): # Added bucket_type default here
        if not self.memory:
            print("MemoryAgent: Memory system not available.")
            return None
        try:
            # memory_type is now handled by VectorMemory.add_memory to be included in tags
            return self.memory.add_memory(embedding, content, tags, source, certainty,
                                         initial_decay_rate, memory_type, bucket_type)
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

    def maintain(self, max_age_days_short_term: int = 7,
                 max_age_days_mid_term: int = 30,
                 max_age_days_long_term: int = 365, # Long term memories also eventually expire
                 short_to_mid_importance_threshold: float = 0.5,
                 mid_to_long_importance_threshold: float = 0.3, # Lower threshold to keep in long-term
                 relevance_boost_factor: float = 0.05,
                 importance_floor: float = 0.1):
        if not self.memory:
            print("MemoryAgent: Memory system not available. Cannot perform maintenance.")
            return

        print("MemoryAgent: Starting memory maintenance cycle (with bucket transitions)...")
        try:
            memories_for_processing = self.memory.get_all_memories_for_decay()
            if not memories_for_processing:
                print("MemoryAgent: No memories found for processing.")
                return

            current_time = time.time()
            updated_count = 0
            deleted_count = 0
            transitioned_count = 0

            for mem_tuple in memories_for_processing:
                # Expected order: id, timestamp, importance, access_count, initial_decay_rate, last_accessed_ts, bucket_type
                memory_data = {
                    'id': mem_tuple[0],
                    'timestamp': mem_tuple[1], # This is a datetime object
                    'importance': mem_tuple[2],
                    'access_count': mem_tuple[3],
                    'initial_decay_rate': mem_tuple[4],
                    'last_accessed_ts': mem_tuple[5], # This is a datetime object
                    'bucket_type': mem_tuple[6]
                }

                # Determine max_age_days based on current bucket for decay calculation
                current_max_age = max_age_days_long_term # Default for long-term or other
                if memory_data['bucket_type'] == 'short_term':
                    current_max_age = max_age_days_short_term
                elif memory_data['bucket_type'] == 'mid_term':
                    current_max_age = max_age_days_mid_term

                new_importance, should_delete = calculate_decayed_importance(
                    memory_data, current_time, current_max_age,
                    relevance_boost_factor, importance_floor
                )

                if should_delete:
                    self.memory.delete_memory(memory_data['id'])
                    deleted_count += 1
                    continue # Skip further processing for this memory

                # Update importance if it changed significantly
                if abs(new_importance - memory_data['importance']) > 1e-5:
                    self.memory.update_memory_importance(memory_data['id'], new_importance)
                    updated_count += 1
                    memory_data['importance'] = new_importance # Use updated importance for transition checks

                # Bucket transition logic (applied after decay/importance update)
                age_days = (current_time - memory_data['timestamp'].timestamp()) / (24 * 3600)

                if memory_data['bucket_type'] == 'short_term':
                    if age_days > max_age_days_short_term or memory_data['importance'] > short_to_mid_importance_threshold:
                        self.memory.update_memory_bucket(memory_data['id'], 'mid_term')
                        transitioned_count += 1
                elif memory_data['bucket_type'] == 'mid_term':
                    if age_days > max_age_days_mid_term or memory_data['importance'] > mid_to_long_importance_threshold:
                        # Only transition if it's important enough for long term, otherwise it will just decay out
                        if memory_data['importance'] > mid_to_long_importance_threshold:
                           self.memory.update_memory_bucket(memory_data['id'], 'long_term')
                           transitioned_count += 1
                    # No transition from long_term in this simple model (they just decay or get deleted)

            self.memory.save() # Checkpoint DB
            print(f"MemoryAgent: Maintenance complete. Updated: {updated_count}, Deleted: {deleted_count}, Transitioned: {transitioned_count}.")

        except Exception as e:
            print(f"MemoryAgent: Error during memory maintenance: {e}")

    def close_memory(self):
        if self.memory:
            print("MemoryAgent: Closing memory connection.")
            self.memory.close()
