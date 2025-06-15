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
        if hasattr(self.db, 'get_memory_by_id'):
             try:
                 return self.db.get_memory_by_id(memory_id)
             except Exception as e:
                 print(f"BioMemoryAgent: Error in get_memory_by_id (via SynapticDuckDB method) for ID {memory_id}: {e}")
                 return None
        else:
            try:
                query = "SELECT * FROM synaptic_memories WHERE neuron_id = ?"
                result = self.db.conn.execute(query, [memory_id]).fetchone()
                if result:
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
        try:
            # Updated to fetch tags as well, needed by IndexerAgent's current batch logic
            query = "SELECT neuron_id, content, tags FROM synaptic_memories"
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
        try:
            # This direct SQL matches what IndexerAgent was trying to do.
            # Assumes 'tags' column exists on synaptic_memories table and is of a list type (e.g., VARCHAR[]).
            # Note: SynapticDuckDB schema defines `tags VARCHAR[]` but was not in `synaptic_memories` table. This needs alignment.
            # For now, assuming the column will be added or this method adapted.
            # If SynapticDuckDB's `synaptic_memories` table doesn't have `tags`, this will fail.
            # The IndexerAgent depends on this.
            # The `synaptic_memories` table in `synaptic_duckdb.py` does NOT have a `tags` column.
            # This method needs to be re-evaluated once `tags` are properly in `synaptic_memories`.
            # For now, let's assume it's being added to SynapticDuckDB's schema.
            # If `tags` is not a list type in DB, list_cat and list_distinct won't work.
            update_sql = "UPDATE synaptic_memories SET tags = list_distinct(list_cat(tags, ?)) WHERE neuron_id = ?"
            # A safer alternative if tags might not exist:
            # update_sql = "UPDATE synaptic_memories SET tags = ? WHERE neuron_id = ?"
            # And then manage the list merging in Python or ensure DB functions handle NULLs.
            self.db.conn.execute(update_sql, [new_tags, memory_id])
        except Exception as e:
            print(f"BioMemoryAgent: Error updating tags for memory ID {memory_id}: {e}")

    def get_memories_for_clustering(self, sample_size: int = None, min_activation_count: int = 0) -> list[dict]:
        '''Fetches memories (neuron_id, memory_trace) for clustering via SynapticDuckDB.'''
        # print(f"BioMemoryAgent: Getting memories for clustering (sample: {sample_size}, min_activations: {min_activation_count})")
        if not self.db:
            print("BioMemoryAgent: SynapticDB not available.")
            return []
        try:
            return self.db.get_memories_for_clustering(sample_size=sample_size, min_activation_count=min_activation_count)
        except Exception as e:
            print(f"BioMemoryAgent: Error getting memories for clustering: {e}")
            return []

    def get_db_connection(self):
        '''Provides direct access to the DuckDB connection for specialized agents if absolutely necessary.'''
        return self.db.conn

    def close(self):
        '''Closes the underlying SynapticDuckDB connection.'''
        print("BioMemoryAgent: Closing SynapticDuckDB connection.")
        if self.db:
            self.db.close()
