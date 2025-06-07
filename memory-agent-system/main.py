import numpy as np
from utils.embedding import get_embedding
from agents.input_agent import InputAgent
from agents.memory_agent import MemoryAgent
from agents.reasoning_agent import ReasoningAgent
from agents.search_agent import SearchAgent # Ensure this class exists
from agents.fusion_agent import FusionAgent # Ensure this class exists
from agents.output_agent import OutputAgent # Ensure this class exists
import time

class AgentOrchestrator:
    def __init__(self):
        self.input_agent = InputAgent()
        self.memory_agent = MemoryAgent()
        self.reasoning_agent = ReasoningAgent()
        self.search_agent = SearchAgent()
        self.fusion_agent = FusionAgent()
        self.output_agent = OutputAgent()

        # Performance metrics
        self.interaction_count = 0

    def process_input(self, user_input: str):
        start_time = time.time()

        # Step 1: Input processing
        input_data = self.input_agent.process(user_input)

        # Step 2: Embedding generation
        # Ensure get_embedding returns a NumPy array as expected by memory_agent
        query_embedding = get_embedding(input_data['content'])
        if not isinstance(query_embedding, np.ndarray):
            # This might happen if get_embedding was changed or if there's an error
            print("Warning: Embedding is not a NumPy array. Attempting conversion.")
            query_embedding = np.array(query_embedding).astype('float32') # Ensure correct dtype

        # Step 3: Memory retrieval
        memory_results = self.memory_agent.retrieve(query_embedding)

        # Step 4: External search if needed
        if input_data['intent'] == 'search':
            search_results = self.search_agent.search(input_data['content'])
        else:
            search_results = []

        # Step 5: Knowledge fusion
        context = self.fusion_agent.fuse(
            input_data['content'], # Changed from user_input to input_data['content'] for consistency
            memory_results,
            search_results
        )

        # Step 6: Reasoning generation
        response = self.reasoning_agent.generate(
            query=input_data['content'],
            context=context
        )

        # Step 7: Store new memory
        self.memory_agent.store(
            embedding=query_embedding, # Storing the original query embedding
            content=f"Q: {input_data['content']}\nA: {response}", # Storing sanitized input
            memory_type="conversation"
        )

        # Step 8: Output delivery
        self.output_agent.deliver(response) # OutputAgent might just print or do nothing

        # Maintenance
        self.interaction_count += 1
        if self.interaction_count % 10 == 0:
            print(f"System: Performing memory maintenance (interaction {self.interaction_count})...")
            self.memory_agent.maintain()

        print(f"System: Processing time: {time.time() - start_time:.2f}s")
        return response

if __name__ == "__main__":
    print("System: Initializing Agent Orchestrator...")
    system = AgentOrchestrator()
    print("System: Multi-Agent System Ready. Type 'exit' to quit.")

    # Initialize logger
    from utils.logger import InteractionLogger # Moved import here to avoid issues if logger fails
    logger = InteractionLogger()

    while True:
        user_input = input("\nUser: ")
        if user_input.lower() in ['exit', 'quit']:
            print("System: Exiting...")
            break

        try:
            response = system.process_input(user_input)
            print(f"Assistant: {response}")
            # Log interaction
            current_context_for_log = system.fusion_agent.fuse( # Re-fuse for logging, or store context
                system.input_agent.process(user_input)['content'],
                system.memory_agent.retrieve(get_embedding(system.input_agent.process(user_input)['content'])),
                [] # Assuming search results are not easily available here for logging, or pass it
            )
            logger.log(user_input, response, current_context_for_log) # Logging disabled for now
        except Exception as e:
            print(f"System: An error occurred: {e}")
            # Optionally, re-initialize parts of the system or provide a fallback
            logger.log(user_input, f"Error: {e}", "N/A") # Logging disabled
