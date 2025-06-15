import time
import numpy as np
import datetime
import pandas as pd

from synaptic_duckdb import SynapticDuckDB
from agents.bio_memory_agent import BioMemoryAgent
from agents.input_agent import InputAgent
from agents.llm_module import LocalLLM
from agents.simple_reasoning_agent import SimpleReasoningAgent
from agents.search_agent import SearchAgent
from agents.fusion_agent import FusionAgent
from agents.output_agent import OutputAgent
from agents.indexer_agent import IndexerAgent # Ensure IndexerAgent is imported
from utils.embedding_utils import get_sentence_embedding, DEFAULT_DIMENSIONS

MAX_REFINEMENT_LOOPS = 2
POSITIVE_REINFORCEMENT_BASE_SIGNAL = 0.25 # Increased slightly
NEGATIVE_REINFORCEMENT_BASE_SIGNAL = -0.15 # Increased slightly

def main_loop():
    print("Initializing Multi-Agent System...")
    synaptic_db = None
    bio_memory_agent = None

    try:
        llm_model_identifier = "EleutherAI/gpt-neo-125M"
        local_llm = LocalLLM(model_name=llm_model_identifier)
        if not local_llm.model: print("CRITICAL: LocalLLM failed to load.")

        synaptic_db = SynapticDuckDB()
        if synaptic_db.dim != DEFAULT_DIMENSIONS: print(f"WARNING: DB dim ({synaptic_db.dim}) != embedding dim ({DEFAULT_DIMENSIONS}).")

        bio_memory_agent = BioMemoryAgent(synaptic_db=synaptic_db)
        input_agent = InputAgent()
        reasoning_agent = SimpleReasoningAgent(local_llm_instance=local_llm)
        search_agent = SearchAgent()
        fusion_agent = FusionAgent()
        output_agent = OutputAgent()
        # Initialize IndexerAgent with a reference to bio_memory_agent and clustering interval
        # e.g., run clustering every 5 times run_indexing_batch is called.
        indexer_agent = IndexerAgent(memory_agent_ref=bio_memory_agent, auto_run_clustering_interval=5)
        print("All components initialized.")

        interaction_count = 0
        consolidation_interval = 5 # Run memory consolidation every 5 interactions
        indexing_batch_interval = 3 # Run IndexerAgent's batch tagging every 3 interactions
        max_mem_items_for_context = 3
        max_search_items_for_context = 2

        while True:
            print("\n------------------------------")
            user_query = input("Ask something (or type 'exit' to quit): ")
            if user_query.lower() == "exit": break

            input_analysis = input_agent.process(user_query)
            query_to_process = input_analysis['processed_content']
            sanitized_full_input = input_analysis['sanitized_input']
            intent = input_analysis['intent']
            requires_web_search_initially = input_analysis['requires_web_search']
            requires_memory_recall_initially = input_analysis['requires_memory_recall']
            # print(f"System: Intent='{input_analysis['intent']}', Query='{query_to_process}', Web={requires_web_search_initially}, Mem={requires_memory_recall_initially}")

            current_query_embedding = get_sentence_embedding(query_to_process)
            memory_context_items = []
            retrieved_memories_df = pd.DataFrame()

            if requires_memory_recall_initially and isinstance(current_query_embedding, list) and len(current_query_embedding) > 0:
                retrieved_memories_df = bio_memory_agent.retrieve_memories(current_query_embedding)
                if not retrieved_memories_df.empty:
                    memory_context_items = retrieved_memories_df.to_dict(orient='records')
                # print(f"System: Initial memory retrieval: {len(memory_context_items)} items.")

            if requires_web_search_initially:
                search_results_list = search_agent.search(query_to_process)
                # print(f"System: Found {len(search_results_list)} initial search results.")

            for loop_count in range(MAX_REFINEMENT_LOOPS):
                # print(f"System: Reasoning attempt {loop_count + 1}/{MAX_REFINEMENT_LOOPS}")
                fused_context, current_used_sources = fusion_agent.fuse(
                    query_to_process, memory_context_items, search_results_list,
                    max_memory_items=max_mem_items_for_context, max_search_items=max_search_items_for_context)
                used_sources_for_output = current_used_sources
                context_for_log = fused_context
                should_check_sufficiency = (loop_count == 0)
                llm_response_data = reasoning_agent.generate(query_to_process, fused_context, check_sufficiency=should_check_sufficiency)

                if llm_response_data.get("status") == "success": break
                elif llm_response_data.get("status") == "needs_more_data":
                    needed_type = llm_response_data.get("type")
                    details_for_new_query = llm_response_data.get("query_details", query_to_process)
                    # print(f"System: Reasoning Agent needs more data. Type: {needed_type}, Details: {details_for_new_query}")
                    if loop_count == MAX_REFINEMENT_LOOPS - 1:
                        llm_response_data = reasoning_agent.generate(query_to_process, fused_context, check_sufficiency=False)
                        break
                    if needed_type == "web_search":
                        new_search_results = search_agent.search(details_for_new_query)
                        search_results_list.extend(new_search_results)
                    elif needed_type == "memory_recall":
                        new_embedding = get_sentence_embedding(details_for_new_query)
                        if isinstance(new_embedding, list) and len(new_embedding) > 0:
                            new_mem_df = bio_memory_agent.retrieve_memories(new_embedding)
                            if not new_mem_df.empty: memory_context_items.extend(new_mem_df.to_dict(orient='records'))
                else: break

            final_answer_text = llm_response_data.get("answer", "Sorry, I could not generate a conclusive answer.")

            output_agent.deliver(final_answer_text, sources=used_sources_for_output)

            feedback_str = "skip"
            if not retrieved_memories_df.empty and 'neuron_id' in retrieved_memories_df.columns and 'relevance' in retrieved_memories_df.columns:
                feedback_str = input("System: Was this helpful? (y/n/skip): ").lower()
                if feedback_str in ['y', 'n']:
                    total_relevance = retrieved_memories_df['relevance'].sum()
                    base_signal = POSITIVE_REINFORCEMENT_BASE_SIGNAL if feedback_str == 'y' else NEGATIVE_REINFORCEMENT_BASE_SIGNAL
                    for _, row in retrieved_memories_df.iterrows():
                        mem_id = row['neuron_id']; mem_relevance = row['relevance']
                        proportional_signal = base_signal * (mem_relevance / total_relevance) if total_relevance > 1e-6 else base_signal / len(retrieved_memories_df)
                        bio_memory_agent.reinforce_memory(mem_id, proportional_signal)
                    # print(f"System: Applied proportional reinforcement to memories.")

            interaction_content = f"Q: {sanitized_full_input}\nA: {final_answer_text}"
            interaction_embedding = get_sentence_embedding(interaction_content)
            interaction_valence = 0.0
            if feedback_str == 'y': interaction_valence = 0.5
            elif feedback_str == 'n': interaction_valence = -0.3

            if isinstance(interaction_embedding, list) and len(interaction_embedding) > 0:
                bio_memory_agent.store_memory(content=interaction_content, embedding=interaction_embedding,
                    emotional_valence=interaction_valence, sensory_modality="interaction_log",
                    temporal_context=datetime.datetime.now())

            interaction_count += 1
            if interaction_count % consolidation_interval == 0:
                bio_memory_agent.run_consolidation_cycle()
            if interaction_count % indexing_batch_interval == 0:
                print(f"\nSystem: Performing periodic indexing (interaction {interaction_count})...")
                indexer_agent.run_indexing_batch() # This will internally decide if it's time to re-cluster

    except KeyboardInterrupt: print("\nSystem: User interrupt. Shutting down...")
    except Exception as e: print(f"System: An unexpected critical error: {e}"); import traceback; traceback.print_exc()
    finally:
        if bio_memory_agent: bio_memory_agent.close()
        elif synaptic_db: synaptic_db.close()
        print("System: Exited.")

if __name__ == "__main__":
    main_loop()
