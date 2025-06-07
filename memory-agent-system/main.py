import numpy as np
from utils.embedding import get_embedding
from agents.input_agent import InputAgent
from agents.memory_agent import MemoryAgent
from agents.reasoning_agent import ReasoningAgent
from agents.search_agent import SearchAgent
from agents.fusion_agent import FusionAgent
from agents.output_agent import OutputAgent
import time
from utils.logger import InteractionLogger # Ensure logger is imported

class AgentOrchestrator:
    def __init__(self):
        print("AgentOrchestrator: Initializing agents...")
        self.input_agent = InputAgent()
        self.memory_agent = MemoryAgent() # Initializes DuckDB connection
        if self.memory_agent.memory is None: # Check if memory init failed
            print("AgentOrchestrator: CRITICAL - MemoryAgent could not initialize memory system. Exiting.")
            raise SystemExit("Memory system initialization failed.")
        self.reasoning_agent = ReasoningAgent()
        if self.reasoning_agent.model is None: # Check if reasoning model init failed
            print("AgentOrchestrator: WARNING - ReasoningAgent could not initialize model. Reasoning will be impaired.")
            # System can continue but reasoning will be affected / use fallback if implemented
        self.search_agent = SearchAgent()
        self.fusion_agent = FusionAgent()
        self.output_agent = OutputAgent()
        self.logger = InteractionLogger() # Initialize logger
        self.interaction_count = 0
        print("AgentOrchestrator: All agents initialized.")

    def process_input(self, user_input: str):
        start_time = time.time()
        response = "Sorry, I encountered an issue processing your request."
        context_for_log = "No context generated due to error."

        try:
            input_data = self.input_agent.process(user_input)
            query_content = input_data['content']
            query_embedding = get_embedding(query_content)

            if not isinstance(query_embedding, np.ndarray) or query_embedding.ndim == 0: # Check if embedding is valid
                print(f"Warning: Embedding for '{query_content}' is not a valid NumPy array or is empty. Skipping memory operations.")
                # Fallback: proceed without memory retrieval or storage if embedding fails
                memory_results = []
            else:
                if query_embedding.ndim == 1:
                     query_embedding = query_embedding.reshape(1, -1) # Ensure 2D for some models, though our store handles 1D from get_embedding
                # The new store expects a 1D array or list, get_embedding provides 1D np.array, which is fine.
                memory_results = self.memory_agent.retrieve(query_embedding)

            search_results = []
            if input_data['intent'] == 'search':
                search_results = self.search_agent.search(query_content)

            context = self.fusion_agent.fuse(
                query_content,
                memory_results,
                search_results
            )
            context_for_log = context # Capture context for logging

            if self.reasoning_agent.model is not None:
                response = self.reasoning_agent.generate(
                    query=query_content,
                    context=context
                )
            else:
                response = "Reasoning module is unavailable. Cannot generate a response."

            # Store new memory (using more metadata fields as per new plan)
            if isinstance(query_embedding, np.ndarray) and query_embedding.ndim > 0:
                 self.memory_agent.store(
                    embedding=query_embedding,
                    content=f"Q: {query_content}\nA: {response}",
                    tags=["conversation", input_data['intent']], # Example tags
                    source="user_interaction",
                    certainty=0.9, # Example certainty for conversation
                    initial_decay_rate=0.96, # Slightly different decay for convos
                    memory_type="conversation_exchange" # More specific type
                )

            self.output_agent.deliver(response)

            self.interaction_count += 1
            if self.interaction_count % 10 == 0:
                print(f"System: Performing memory maintenance (interaction {self.interaction_count})...")
                self.memory_agent.maintain() # This now uses DuckDB and correct decay

        except Exception as e:
            print(f"AgentOrchestrator: Error in process_input: {e}")
            # response is already set to a default error message
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
