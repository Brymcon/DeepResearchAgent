import time
import numpy as np
import datetime # Though NeuroplasticMind handles datetime.datetime.now() internally
from neuroplastic_mind import NeuroplasticMind
# Ensure synaptic_duckdb.py (imported by neuroplastic_mind.py) is in the same directory or Python path

DIMENSIONS = 384 # Should match the default in SynapticDuckDB or be configurable

def get_dummy_embedding(text: str, dim: int = DIMENSIONS) -> list[float]:
    """Generates a dummy embedding from text by hashing and normalizing."""
    # Simple hash-based embedding for consistent but dummy vectors
    if not text: return [0.0] * dim
    seed = sum(ord(c) for c in text)
    rng = np.random.RandomState(seed)
    embedding = rng.rand(dim).astype(np.float32)
    norm = np.linalg.norm(embedding)
    if norm == 0: return [0.0] * dim
    return (embedding / norm).tolist()

def generate_mock_experience(idx: int) -> dict:
    content = f"This is mock experience number {idx}. It contains some unique words like Xylarion{idx}."
    experience = {
        "content": content,
        "embedding": get_dummy_embedding(content),
        "valence": np.random.uniform(-0.8, 0.8), # Random emotional valence
        "modality": "text"
    }
    if idx % 3 == 0: # Add some feedback for some experiences
        experience["feedback"] = {
            "strength": np.random.uniform(0.3, 0.9),
            "insight": f"Generated insight from experience {idx}",
            "correction": {"layer1": [np.random.uniform(-0.1,0.1) for _ in range(128)]} # Example correction
        }
    return experience

if __name__ == "__main__":
    print("Initializing NeuroplasticMind...")
    # Placeholder path - CorticalColumn will handle failure if model doesn't exist
    # User should replace with an actual path to a GGUF model for CorticalColumn to load.
    gguf_model_path = "./models/placeholder_model.gguf"
    # Ensure the directory for the DB exists, SynapticDuckDB also does this
    # os.makedirs("memory_db", exist_ok=True) # SynapticDuckDB handles its own DB dir

    brain = None
    try:
        brain = NeuroplasticMind(
            base_model_path=gguf_model_path,
            growth_plan="infant",
            # synaptic_db_path="memory_db/test_neuro_memory.duckdb" # Optional: specify a test DB path
        )
        print("NeuroplasticMind initialized.")

        num_experiences_to_simulate = 150 # Simulate enough experiences to potentially trigger maturation/consolidation
        experiences_processed_this_cycle = 0

        for i in range(num_experiences_to_simulate):
            print(f"\n--- Simulating Experience {i+1} ---")
            experience = generate_mock_experience(i)

            # 1. Perceive and Encode
            perception_result = brain.perceive(sensory_input=experience)
            memory_id = perception_result.get("memory_id")
            print(f"  Perceived and encoded. Memory ID: {memory_id}")

            # 2. Learn from feedback (if any)
            if experience.get('feedback') and memory_id is not None:
                print(f"  Applying learning feedback for memory ID: {memory_id}")
                brain.learn(
                    feedback=experience['feedback'],
                    memory_id=memory_id
                )

            # 3. Simple Recall example (e.g., recall based on current experience's embedding)
            if i % 5 == 0 and memory_id is not None: # Every 5 experiences, try a recall
                print(f"  Attempting recall based on current experience embedding...")
                recalled_memories_df = brain.recall(cue_embedding=experience['embedding'], depth=2)
                if not recalled_memories_df.empty:
                    print(f"    Recalled {len(recalled_memories_df)} memories. Top one content: ... {recalled_memories_df.iloc[0]['content'][:50]} ...")
                else:
                    print("    No memories recalled for this cue.")

            experiences_processed_this_cycle += 1
            current_stage_info = brain.growth_stages[brain.current_stage]
            consolidation_interval = current_stage_info.get("consolidation_interval_exp", 100) # Default if not set

            # 4. Periodic Consolidation (Dream Cycle)
            if experiences_processed_this_cycle >= consolidation_interval:
                print(f"\n--- Triggering Dream Cycle (after {experiences_processed_this_cycle} experiences) ---")
                brain.dream_cycle()
                experiences_processed_this_cycle = 0 # Reset counter for this cycle

            # Maturation is handled internally by _encode_memory -> _should_mature -> mature
            # We can print current stage to observe if it changes
            if (i + 1) % 25 == 0: # Print stage periodically
                print(f"  Status Check: Current Stage = {brain.current_stage}, Experience Count in Stage = {brain.experience_counter}")

            time.sleep(0.05) # Small delay to make output readable

        print("\n--- Simulation Complete ---")

    except Exception as e:
        print(f"An error occurred during the simulation: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if brain and hasattr(brain, 'synapse_db') and brain.synapse_db:
            print("Closing SynapticDuckDB connection...")
            brain.synapse_db.close()
        print("Exiting main_bio.")
