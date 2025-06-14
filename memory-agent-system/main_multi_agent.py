import time
import numpy as np
import datetime
import pandas as pd # For handling DataFrame from recall

from synaptic_duckdb import SynapticDuckDB
from agents.bio_memory_agent import BioMemoryAgent
from agents.input_agent import InputAgent # Using the enhanced one
from agents.llm_module import LocalLLM
from agents.simple_reasoning_agent import SimpleReasoningAgent
from agents.search_agent import SearchAgent # Using the enhanced one
from agents.fusion_agent import FusionAgent # Using the enhanced one
from agents.output_agent import OutputAgent # Using the enhanced one
# from agents.indexer_agent import IndexerAgent # Will be integrated later

DIMENSIONS = 384 # Default embedding dimension

def get_embedding(text: str, dim: int = DIMENSIONS) -> list[float]:
    """Generates a dummy embedding. Replace with actual sentence transformer.
       IMPORTANT: This is a DUMMY function. For real use, integrate a sentence embedding model.
    """
    if not text or not isinstance(text, str):
        # print(f"Warning: Invalid input to get_embedding: {text}. Returning zero vector.")
        return [0.0] * dim
    seed = sum(ord(c) for c in text)
    rng = np.random.RandomState(seed)
    embedding = rng.rand(dim).astype(np.float32)
    norm = np.linalg.norm(embedding)
    if norm == 0:
        return [0.0] * dim
    return (embedding / norm).tolist()

def main_loop():
    print("Initializing Multi-Agent System with BioMemory and LocalLLM...")
    synaptic_db = None
    bio_memory_agent = None

    try:
        # --- Initialize Core Components ---
        # Specify a GGUF model path or use a HuggingFace model name for LocalLLM
        # For testing without a large download, EleutherAI/gpt-neo-125M is small.
        # User should replace with their desired model, e.g., a Llama3 GGUF path.
        llm_model_identifier = "EleutherAI/gpt-neo-125M" # Fallback/Test model
        # llm_model_identifier = "path/to/your/local/model.gguf"

        local_llm = LocalLLM(model_name=llm_model_identifier) # or model_path= for GGUF
        if not local_llm.model:
            print("CRITICAL: LocalLLM failed to load. Some functionalities will be impaired or disabled.")
            # Decide if system should exit or run in a very limited mode
            # For now, it will run but reasoning will fail.

        synaptic_db = SynapticDuckDB() # Uses default DB path
        bio_memory_agent = BioMemoryAgent(synaptic_db=synaptic_db)

        # --- Initialize Agents ---
        input_agent = InputAgent()
        reasoning_agent = SimpleReasoningAgent(local_llm_instance=local_llm)
        search_agent = SearchAgent() # Uses env vars for API key or mock data
        fusion_agent = FusionAgent()
        output_agent = OutputAgent()
        # indexer_agent = IndexerAgent(bio_memory_agent) # For later integration

        print("All components initialized.")

        interaction_count = 0
        consolidation_interval = 5 # Run consolidation every 5 interactions for testing

        # --- Main Interaction Loop ---
        while True:
            print("\n------------------------------")
            user_query = input("Ask something (or type 'exit' to quit): ")
            if user_query.lower() == "exit":
                break

            # 1. Process Input
            input_analysis = input_agent.process(user_query)
            processed_query_content = input_analysis['processed_content']
            sanitized_full_input = input_analysis['sanitized_input']
            intent = input_analysis['intent']
            print(f"System: Intent='{intent}', Processed Query='{processed_query_content}'")

            # 2. Retrieve Memories
            query_embedding = get_embedding(processed_query_content)
            retrieved_memories_df = bio_memory_agent.retrieve_memories(query_embedding)
            memory_context_items = []
            if not retrieved_memories_df.empty:
                memory_context_items = retrieved_memories_df.to_dict(orient='records')
            print(f"System: Retrieved {len(memory_context_items)} memories.")

            # 3. Search (if needed)
            search_results_list = []
            if intent.startswith("search"): # e.g., 'search_capital', 'search_general'
                print(f"System: Performing web search for '{processed_query_content}'...")
                search_results_list = search_agent.search(processed_query_content)
                print(f"System: Found {len(search_results_list)} search results.")

            # 4. Fuse Context
            fused_context = fusion_agent.fuse(processed_query_content, memory_context_items, search_results_list)
            # print(f"System: Fused context (first 200 chars): {fused_context[:200]}...")

            # 5. Generate Response
            prompt = f"Based on the following context, answer the question.\n\nContext:\n{fused_context}\n\nQuestion: {processed_query_content}\nAnswer:"
            response_text = reasoning_agent.generate(prompt)

            # 6. Deliver Response
            output_agent.deliver(response_text)

            # 7. Self-Improvement Loop (Feedback)
            if not retrieved_memories_df.empty: # Only ask for feedback if memories were used
                feedback_str = input("System: Was this helpful? (y/n/skip): ").lower()
                if feedback_str == 'y':
                    for mem_id in retrieved_memories_df['neuron_id']:
                        bio_memory_agent.reinforce_memory(mem_id, 0.2) # Positive reinforcement signal
                    print("System: Thanks! I've strengthened the memories that helped.")
                elif feedback_str == 'n':
                    for mem_id in retrieved_memories_df['neuron_id']:
                        bio_memory_agent.reinforce_memory(mem_id, -0.1) # Negative reinforcement signal
                    print("System: Noted. I've adjusted the relevance of those memories.")

            # 8. Store Interaction as New Memory
            # Use the full sanitized input for the question part of the stored memory
            interaction_content = f"Q: {sanitized_full_input}\nA: {response_text}"
            interaction_embedding = get_embedding(interaction_content) # Embed the whole Q&A

            # Determine valence based on feedback if possible
            interaction_valence = 0.0
            if feedback_str == 'y': interaction_valence = 0.5
            elif feedback_str == 'n': interaction_valence = -0.3

            bio_memory_agent.store_memory(
                content=interaction_content,
                embedding=interaction_embedding,
                emotional_valence=interaction_valence,
                sensory_modality="interaction_log",
                temporal_context=datetime.datetime.now()
            )
            print("System: Stored current interaction as a new memory.")

            # 9. Periodic Maintenance
            interaction_count += 1
            if interaction_count % consolidation_interval == 0:
                print(f"\nSystem: Performing periodic memory consolidation (interaction {interaction_count})...")
                bio_memory_agent.run_consolidation_cycle()
            # Add IndexerAgent call here when ready:
            # if interaction_count % indexing_interval == 0: indexer_agent.run_indexing_batch()

    except KeyboardInterrupt:
        print("\nSystem: User interrupt detected. Shutting down...")
    except Exception as e:
        print(f"System: An unexpected critical error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if bio_memory_agent:
            print("System: Closing BioMemoryAgent and SynapticDuckDB connection...")
            bio_memory_agent.close()
        elif synaptic_db: # If only DB was init'd
            synaptic_db.close()
        print("System: Exited.")

if __name__ == "__main__":
    main_loop()
