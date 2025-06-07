import numpy as np
from utils.embedding import get_embedding
from agents.input_agent import InputAgent
from agents.memory_agent import MemoryAgent
from agents.reasoning_agent import ReasoningAgent
from agents.search_agent import SearchAgent
from agents.fusion_agent import FusionAgent
from agents.output_agent import OutputAgent
from agents.indexer_agent import IndexerAgent # Import IndexerAgent
import time
from utils.logger import InteractionLogger

class AgentOrchestrator:
    def __init__(self):
        print("AgentOrchestrator: Initializing agents...")
        self.input_agent = InputAgent()
        self.memory_agent = MemoryAgent()
        if self.memory_agent.memory is None:
            print("AgentOrchestrator: CRITICAL - MemoryAgent could not initialize memory system. Exiting.")
            raise SystemExit("Memory system initialization failed.")

        self.indexer_agent = IndexerAgent(memory_agent_ref=self.memory_agent) # Initialize IndexerAgent

        self.reasoning_agent = ReasoningAgent()
        if self.reasoning_agent.model is None:
            print("AgentOrchestrator: WARNING - ReasoningAgent could not initialize model. Reasoning will be impaired.")
        self.search_agent = SearchAgent()
        self.fusion_agent = FusionAgent()
        self.output_agent = OutputAgent()
        self.logger = InteractionLogger()
        self.interaction_count = 0

        self.maintenance_interval = 10 # interactions for memory decay
        self.indexing_interval = 20 # interactions for running indexer batch (less frequent)
        self.maintenance_params = {
            'max_age_days_short_term': 7,
            'max_age_days_mid_term': 30,
            'max_age_days_long_term': 365,
            'short_to_mid_importance_threshold': 0.5,
            'mid_to_long_importance_threshold': 0.3,
            'relevance_boost_factor': 0.05,
            'importance_floor': 0.1
        }
        print("AgentOrchestrator: All agents initialized.")

    def process_input(self, user_input: str):
        start_time = time.time()
        response = "Sorry, I encountered an issue processing your request."
        context_for_log = "No context generated due to error."

        try:
            input_data = self.input_agent.process(user_input)
            query_content = input_data['content']
            query_embedding = get_embedding(query_content)

            memory_results = []
            if isinstance(query_embedding, np.ndarray) and query_embedding.size > 0:
                memory_results = self.memory_agent.retrieve(query_embedding)
            else:
                print(f"Warning: Embedding for '{query_content}' is not valid. Skipping memory retrieval.")

            search_results = []
            if input_data['intent'] == 'search':
                search_results = self.search_agent.search(query_content)

            context = self.fusion_agent.fuse(
                query_content,
                memory_results,
                search_results
            )
            context_for_log = context

            if self.reasoning_agent.model is not None:
                response = self.reasoning_agent.generate(
                    query=query_content,
                    context=context
                )
            else:
                response = "Reasoning module is unavailable. Cannot generate a response."

            if isinstance(query_embedding, np.ndarray) and query_embedding.size > 0:
                 memory_id = self.memory_agent.store(
                    embedding=query_embedding,
                    content=f"Q: {query_content}\nA: {response}",
                    tags=["conversation", input_data['intent']],
                    source="user_interaction",
                    certainty=0.9,
                    initial_decay_rate=0.96,
                    memory_type="conversation_exchange",
                    bucket_type='short_term'
                )
                 # Optionally, trigger single item indexing immediately if memory_id is returned and valid
                 # if memory_id is not None:
                 #    self.indexer_agent.process_memory_entry_for_indexing(memory_id)
                 # For now, relying on batch indexing below.

            self.output_agent.deliver(response)

            self.interaction_count += 1
            if self.interaction_count % self.maintenance_interval == 0:
                print(f"System: Performing memory maintenance (interaction {self.interaction_count})...")
                self.memory_agent.maintain(**self.maintenance_params)

            if self.interaction_count % self.indexing_interval == 0:
                print(f"System: Performing batch keyword indexing (interaction {self.interaction_count})...")
                self.indexer_agent.run_indexing_batch()

        except Exception as e:
            print(f"AgentOrchestrator: Error in process_input: {e}")
            context_for_log = f"Error occurred: {e}"
        finally:
            processing_time = time.time() - start_time
            print(f"System: Processing time: {processing_time:.2f}s")
            try:
                self.logger.log(user_input, response, context_for_log)
            except Exception as log_e:
                print(f"AgentOrchestrator: Error logging interaction: {log_e}")
        return response

    def shutdown(self):
        print("AgentOrchestrator: Shutting down...")
        if self.memory_agent:
            self.memory_agent.close_memory()
        print("AgentOrchestrator: Shutdown complete.")


if __name__ == "__main__":
    system = None
    try:
        print("System: Initializing Agent Orchestrator...")
        system = AgentOrchestrator()
        print("System: Multi-Agent System Ready. Type 'exit' or 'quit' to quit.")

        while True:
            user_input = input("\nUser: ")
            if user_input.lower() in ['exit', 'quit']:
                print("System: Exiting...")
                break
            response = system.process_input(user_input)
            print(f"Assistant: {response}")

    except SystemExit as e:
        print(f"System: Exiting due to critical error: {e}")
    except KeyboardInterrupt:
        print("\nSystem: Keyboard interrupt detected. Exiting...")
    except Exception as e:
        print(f"System: An unexpected critical error occurred: {e}")
    finally:
        if system:
            system.shutdown()
