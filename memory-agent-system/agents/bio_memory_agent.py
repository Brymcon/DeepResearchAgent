# Agent that interfaces with the SynapticDuckDB memory core

import numpy as np
import datetime
# We expect SynapticDuckDB to be in a place where it can be imported,
# e.g., same directory or memory-agent-system is in PYTHONPATH
# For development, assuming it's in memory-agent-system/synaptic_duckdb.py
from synaptic_duckdb import SynapticDuckDB # This line might need adjustment based on final project structure

class BioMemoryAgent:
    def __init__(self, synaptic_db: SynapticDuckDB):
        '''
        Initializes the BioMemoryAgent with an instance of SynapticDuckDB.
        Args:
            synaptic_db: An initialized instance of the SynapticDuckDB class.
        '''
        if not isinstance(synaptic_db, SynapticDuckDB):
            raise TypeError("BioMemoryAgent requires a SynapticDuckDB instance.")
        self.db = synaptic_db
        print("BioMemoryAgent: Initialized with SynapticDuckDB instance.")

    def store_memory(self, content: str, embedding: list[float],
                     emotional_valence: float, sensory_modality: str,
                     temporal_context: datetime.datetime = None) -> int | None:
        '''Stores a new memory item via SynapticDuckDB.'''
        if temporal_context is None:
            temporal_context = datetime.datetime.now()
        # print(f"BioMemoryAgent: Storing memory - Content: {content[:50]}...")
        try:
            return self.db.store_memory(
                content=content,
                embedding=embedding,
                emotional_valence=emotional_valence,
                sensory_modality=sensory_modality,
                temporal_context=temporal_context
            )
        except Exception as e:
            print(f"BioMemoryAgent: Error in store_memory: {e}")
            return None

    def retrieve_memories(self, cue_embedding: list[float], depth: int = 3,
                          temporal_decay_rate_hourly: float = 0.05,
                          similarity_threshold_initial: float = 0.6,
                          result_limit: int = 10) -> 'pd.DataFrame': # Forward reference for DataFrame
        '''Retrieves memories using associative recall from SynapticDuckDB.'''
        # print(f"BioMemoryAgent: Retrieving memories with cue_embedding (first 5 dims): {cue_embedding[:5]}...")
        try:
            return self.db.associative_recall(
                cue_embedding=cue_embedding,
                depth=depth,
                temporal_decay_rate_hourly=temporal_decay_rate_hourly,
                similarity_threshold_initial=similarity_threshold_initial,
                result_limit=result_limit
            )
        except Exception as e:
            print(f"BioMemoryAgent: Error in retrieve_memories: {e}")
            import pandas as pd # Ensure pandas is available for empty DF
            return pd.DataFrame() # Return empty DataFrame on error

    def reinforce_memory(self, memory_id: int, reinforcement_signal: float, **kwargs):
        '''Reinforces a specific memory in SynapticDuckDB.'''
        # print(f"BioMemoryAgent: Reinforcing memory ID {memory_id} with signal {reinforcement_signal}.")
        try:
            self.db.reinforce_memory(memory_id, reinforcement_signal, **kwargs)
        except Exception as e:
            print(f"BioMemoryAgent: Error in reinforce_memory for ID {memory_id}: {e}")

    def run_consolidation_cycle(self, pruning_threshold: float = 0.2,
                                cross_link_sim_thresh: float = 0.65,
                                cross_link_valence_diff: float = 0.3,
                                cross_link_init_strength: float = 0.4):
        '''Runs a memory consolidation cycle involving pruning and cross-linking.'''
        print("BioMemoryAgent: Running memory consolidation cycle...")
        try:
            self.db.synaptic_pruning(threshold=pruning_threshold)
            self.db.cross_link_modalities(
                similarity_threshold=cross_link_sim_thresh,
                valence_diff_threshold=cross_link_valence_diff,
                initial_strength=cross_link_init_strength
            )
            print("BioMemoryAgent: Consolidation cycle complete.")
        except Exception as e:
            print(f"BioMemoryAgent: Error during consolidation cycle: {e}")

    def remap_connections_for_maturation(self, new_association_depth: int, initial_strength: float = 0.35):
        '''Triggers connection remapping in SynapticDuckDB for developmental maturation.'''
        print(f"BioMemoryAgent: Remapping connections for new association depth: {new_association_depth}.")
        try:
            self.db.remap_connections(new_association_depth, initial_strength)
        except Exception as e:
            print(f"BioMemoryAgent: Error during connection remapping: {e}")

    def get_memory_by_id(self, memory_id: int) -> dict | None:
        '''Retrieves a single memory by its ID via SynapticDuckDB.'''
        # This method might be needed by IndexerAgent or other specific agents
        # The get_memory_by_id was defined in old VectorMemory, useful to have here too.
        # SynapticDuckDB doesn't have this method explicitly yet, we can add it or call SQL directly.
        # For now, let's assume SynapticDuckDB should have it or we add a direct query here.
        # print(f"BioMemoryAgent: Getting memory by ID {memory_id}")
        if hasattr(self.db, 'get_memory_by_id'): # If SynapticDuckDB implements it
             try:
                 return self.db.get_memory_by_id(memory_id)
             except Exception as e:
                 print(f"BioMemoryAgent: Error in get_memory_by_id (via SynapticDuckDB method) for ID {memory_id}: {e}")
                 return None
        else:
            # print("BioMemoryAgent: SynapticDuckDB does not have get_memory_by_id, direct query placeholder.")
            # Placeholder: direct query if SynapticDuckDB doesn't have the method
            try:
                query = "SELECT * FROM synaptic_memories WHERE neuron_id = ?"
                result = self.db.conn.execute(query, [memory_id]).fetchone()
                if result:
                    # Convert tuple to dict - assumes column order or get column names
                    # This part needs to be robust if used
                    cols = [desc[0] for desc in self.db.conn.description] if self.db.conn.description else []
                    return dict(zip(cols, result)) if cols else None
                return None
            except Exception as e:
                print(f"BioMemoryAgent: Error in get_memory_by_id (direct query) for ID {memory_id}: {e}")
                return None

    def get_all_memories_for_processing(self, limit: int = None) -> list[dict]:
        '''Fetches all memories or a subset, e.g., for IndexerAgent.
           Returns list of dicts.
        '''
        # print(f"BioMemoryAgent: Getting all memories for processing (limit: {limit})")
        # This method might be needed by IndexerAgent. SynapticDuckDB doesn't have this explicitly.
        # We can add it to SynapticDuckDB or query directly here.
        try:
            query = "SELECT neuron_id, content, tags FROM synaptic_memories" # Example fields for Indexer
            if limit:
                query += f" LIMIT {int(limit)}"
            results_raw = self.db.conn.execute(query).fetchall()
            memories = []
            cols = [desc[0] for desc in self.db.conn.description] if self.db.conn.description else []
            if cols:
                for row in results_raw:
                    memories.append(dict(zip(cols, row)))
            return memories
        except Exception as e:
            print(f"BioMemoryAgent: Error in get_all_memories_for_processing: {e}")
            return []

    def update_memory_tags(self, memory_id: int, new_tags: list[str]):
        '''Updates the tags for a given memory. Used by IndexerAgent.'''
        # print(f"BioMemoryAgent: Updating tags for memory ID {memory_id}")
        # This method is needed by IndexerAgent. Assumes direct execution or method in SynapticDuckDB.
        try:
            # Using list_distinct(list_cat(tags, new_tags_param)) might be too simple if we want to *replace* or *ensure only new* tags.
            # For now, let's assume a full replacement or a more sophisticated merge in SynapticDuckDB if needed.
            # The IndexerAgent currently uses: list_distinct(list_cat(tags, ?))
            # So, this BioMemoryAgent method should just pass it through if SynapticDuckDB handles it,
            # or implement the list_distinct(list_cat(tags, ?)) here directly.
            # For now, direct SQL as IndexerAgent does it.
            # The IndexerAgent actually calls self.memory_agent.memory.conn.execute directly.
            # This should be refactored so IndexerAgent calls a method on BioMemoryAgent.
            update_sql = "UPDATE synaptic_memories SET tags = list_distinct(list_cat(tags, ?)) WHERE neuron_id = ?"
            self.db.conn.execute(update_sql, [new_tags, memory_id])
        except Exception as e:
            print(f"BioMemoryAgent: Error updating tags for memory ID {memory_id}: {e}")

    def get_db_connection(self):
        '''Provides direct access to the DuckDB connection for specialized agents if absolutely necessary.'''
        return self.db.conn

    def close(self):
        '''Closes the underlying SynapticDuckDB connection.'''
        print("BioMemoryAgent: Closing SynapticDuckDB connection.")
        if self.db:
            self.db.close()
