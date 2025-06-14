import time
import numpy as np
import datetime

from synaptic_duckdb import SynapticDuckDB
from agents.bio_memory_agent import BioMemoryAgent
# Import other agents as they are integrated
# from agents.input_agent import InputAgent
# from agents.reasoning_agent import ReasoningAgent
# etc.

DIMENSIONS = 384 # Should match the default in SynapticDuckDB or be configurable

def get_dummy_embedding(text: str, dim: int = DIMENSIONS) -> list[float]:
    """Generates a dummy embedding from text by hashing and normalizing."""
    if not text: return [0.0] * dim
    seed = sum(ord(c) for c in text)
    rng = np.random.RandomState(seed)
    embedding = rng.rand(dim).astype(np.float32)
    norm = np.linalg.norm(embedding)
    if norm == 0: return [0.0] * dim
    return (embedding / norm).tolist()

if __name__ == "__main__":
    print("Initializing Multi-Agent System with BioMemory...")
    synaptic_db = None
    bio_memory_agent = None

    try:
        # Initialize SynapticDuckDB (persistent)
        # db_path = "memory_db/multi_agent_neuro_memory.duckdb" # Example of a different DB file
        synaptic_db = SynapticDuckDB() # Uses default path from synaptic_duckdb.py
        print("SynapticDuckDB initialized.")

        # Initialize BioMemoryAgent with the SynapticDuckDB instance
        bio_memory_agent = BioMemoryAgent(synaptic_db=synaptic_db)
        print("BioMemoryAgent initialized.")

        print("\n--- Testing BioMemoryAgent ---  ")
        # Test 1: Store a memory
        print("Attempting to store a test memory...")
        test_content = "This is a test memory for the multi-agent system."
        test_embedding = get_dummy_embedding(test_content)
        test_valence = 0.5
        test_modality = "text_test"

        stored_id = bio_memory_agent.store_memory(
            content=test_content,
            embedding=test_embedding,
            emotional_valence=test_valence,
            sensory_modality=test_modality,
            temporal_context=datetime.datetime.now()
        )
        if stored_id is not None:
            print(f"Test memory stored successfully. Neuron ID: {stored_id}")

            # Test 2: Retrieve the memory
            print(f"Attempting to retrieve memories similar to the test memory...")
            retrieved_df = bio_memory_agent.retrieve_memories(cue_embedding=test_embedding, k=3)
            if not retrieved_df.empty:
                print(f"Retrieved {len(retrieved_df)} memories. Top result:")
                print(retrieved_df.head(1))
            else:
                print("No memories retrieved, or an error occurred.")

            # Test 3: Reinforce the memory
            print(f"Attempting to reinforce memory {stored_id}...")
            bio_memory_agent.reinforce_memory(memory_id=stored_id, reinforcement_signal=0.8)
            print(f"Reinforcement call completed for memory {stored_id}.")
            # We could fetch it again to see if synaptic_strength changed, but SynapticDuckDB prints this.
        else:
            print("Failed to store test memory.")

        # Test 4: Run consolidation cycle
        print("Attempting to run a memory consolidation cycle...")
        bio_memory_agent.run_consolidation_cycle()
        print("Consolidation cycle call completed.")

        print("\n--- BioMemoryAgent Test Complete --- ")
        print("Further agent integration will happen in Phase 2.")

    except Exception as e:
        print(f"An error occurred during initialization or testing: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if bio_memory_agent:
            print("Closing BioMemoryAgent (and SynapticDuckDB connection)...")
            bio_memory_agent.close()
        elif synaptic_db: # If bio_memory_agent failed to init but db was created
            print("Closing SynapticDuckDB connection directly...")
            synaptic_db.close()
        print("Exiting main_multi_agent.")
