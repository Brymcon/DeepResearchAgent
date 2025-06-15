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
from utils.embedding_utils import get_sentence_embedding, DEFAULT_DIMENSIONS
# from agents.indexer_agent import IndexerAgent

MAX_REFINEMENT_LOOPS = 2 # Max times to try fetching more data for reasoning

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
        print("All components initialized.")

        interaction_count = 0
        consolidation_interval = 5
        max_mem_items_for_context = 3
        max_search_items_for_context = 2

        while True:
            print("\n------------------------------")
            user_query = input("Ask something (or type 'exit' to quit): ")
            if user_query.lower() == "exit": break

            input_analysis = input_agent.process(user_query)
            query_to_process = input_analysis['processed_content']
            sanitized_full_input = input_analysis['sanitized_input']
            # intent = input_analysis['intent'] # Not directly used for search trigger anymore
            requires_web_search_initially = input_analysis['requires_web_search']
            requires_memory_recall_initially = input_analysis['requires_memory_recall']
            print(f"System: Initial Query='{query_to_process}', WebSearchFlag={requires_web_search_initially}, MemRecallFlag={requires_memory_recall_initially}")

            current_query_embedding = get_sentence_embedding(query_to_process)
            memory_context_items = []
            retrieved_memories_df = pd.DataFrame()
            search_results_list = []
            llm_response_data = {"status": "error", "answer": "Initial error before processing.", "reasoning": ""}
            used_sources_for_output = []
            context_for_log = "No context generated."

            # Initial data gathering based on InputAgent flags
            if requires_memory_recall_initially and isinstance(current_query_embedding, list) and len(current_query_embedding) > 0:
                retrieved_memories_df = bio_memory_agent.retrieve_memories(current_query_embedding)
                if not retrieved_memories_df.empty:
                    memory_context_items = retrieved_memories_df.to_dict(orient='records')
                print(f"System: Initial memory retrieval: {len(memory_context_items)} items.")

            if requires_web_search_initially:
                print(f"System: Initial web search for '{query_to_process}'...")
                search_results_list = search_agent.search(query_to_process)
                print(f"System: Found {len(search_results_list)} initial search results.")

            # Iterative Refinement Loop
            for loop_count in range(MAX_REFINEMENT_LOOPS):
                print(f"System: Reasoning attempt {loop_count + 1}/{MAX_REFINEMENT_LOOPS}")
                fused_context, current_used_sources = fusion_agent.fuse(
                    query_to_process, memory_context_items, search_results_list,
                    max_memory_items=max_mem_items_for_context,
                    max_search_items=max_search_items_for_context
                )
                used_sources_for_output = current_used_sources # Update sources for output
                context_for_log = fused_context

                # In the first loop, check sufficiency. In subsequent loops, go straight to answer.
                should_check_sufficiency = (loop_count == 0)
                llm_response_data = reasoning_agent.generate(query_to_process, fused_context, check_sufficiency=should_check_sufficiency)

                if llm_response_data.get("status") == "success":
                    break # Sufficient context, got an answer
                elif llm_response_data.get("status") == "needs_more_data":
                    needed_type = llm_response_data.get("type")
                    details_for_new_query = llm_response_data.get("query_details", query_to_process)
                    print(f"System: Reasoning Agent needs more data. Type: {needed_type}, Details: {details_for_new_query}")
                    if loop_count == MAX_REFINEMENT_LOOPS - 1: # Last attempt
                        print("System: Max refinement loops reached. Proceeding with current information.")
                        # Attempt to generate final response without sufficiency check
                        llm_response_data = reasoning_agent.generate(query_to_process, fused_context, check_sufficiency=False)
                        break

                    if needed_type == "web_search":
                        print(f"System: Performing additional web search for '{details_for_new_query}'...")
                        new_search_results = search_agent.search(details_for_new_query)
                        print(f"System: Found {len(new_search_results)} additional search results.")
                        search_results_list.extend(new_search_results) # Append new results
                    elif needed_type == "memory_recall":
                        print(f"System: Performing additional memory recall for '{details_for_new_query}'...")
                        new_embedding = get_sentence_embedding(details_for_new_query)
                        if isinstance(new_embedding, list) and len(new_embedding) > 0:
                            new_mem_df = bio_memory_agent.retrieve_memories(new_embedding)
                            if not new_mem_df.empty:
                                print(f"System: Found {len(new_mem_df)} additional memories.")
                                memory_context_items.extend(new_mem_df.to_dict(orient='records'))
                        else:
                            print("System: Invalid embedding for additional memory recall.")
                else: # Error or unexpected status
                    print(f"System: Reasoning agent returned status: {llm_response_data.get('status')}. Using current answer.")
                    break

            final_answer_text = llm_response_data.get("answer", "Sorry, I could not generate a conclusive answer.")
            # final_reasoning_text = llm_response_data.get("reasoning", "") # For future use by OutputAgent

            output_agent.deliver(final_answer_text, sources=used_sources_for_output)

            feedback_str = "skip"
            if not retrieved_memories_df.empty:
                feedback_str = input("System: Was this helpful? (y/n/skip): ").lower()
                if feedback_str == 'y':
                    for mem_id in retrieved_memories_df['neuron_id']:
                        bio_memory_agent.reinforce_memory(mem_id, 0.2)
                    print("System: Strengthened helpful memories.")
                elif feedback_str == 'n':
                    for mem_id in retrieved_memories_df['neuron_id']:
                        bio_memory_agent.reinforce_memory(mem_id, -0.1)
                    print("System: Adjusted unhelpful memories.")

            interaction_content = f"Q: {sanitized_full_input}\nA: {final_answer_text}"
            interaction_embedding = get_sentence_embedding(interaction_content)
            interaction_valence = 0.0
            if feedback_str == 'y': interaction_valence = 0.5
            elif feedback_str == 'n': interaction_valence = -0.3

            if isinstance(interaction_embedding, list) and len(interaction_embedding) > 0:
                bio_memory_agent.store_memory(
                    content=interaction_content, embedding=interaction_embedding,
                    emotional_valence=interaction_valence, sensory_modality="interaction_log",
                    temporal_context=datetime.datetime.now()
                )
            else: print("Warning: Invalid embedding for interaction log. Skipping storage.")

            interaction_count += 1
            if interaction_count % consolidation_interval == 0:
                bio_memory_agent.run_consolidation_cycle()

    except KeyboardInterrupt: print("\nSystem: User interrupt. Shutting down...")
    except Exception as e: print(f"System: An unexpected critical error: {e}"); import traceback; traceback.print_exc()
    finally:
        if bio_memory_agent: bio_memory_agent.close()
        elif synaptic_db: synaptic_db.close()
        print("System: Exited.")

if __name__ == "__main__":
    main_loop()
