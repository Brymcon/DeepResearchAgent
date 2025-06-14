from .llm_module import LocalLLM # Assuming llm_module is in the same directory

class SimpleReasoningAgent:
    def __init__(self, local_llm_instance: LocalLLM):
        """
        Initializes the SimpleReasoningAgent with an instance of LocalLLM.
        Args:
            local_llm_instance: An initialized instance of the LocalLLM class.
        """
        if not isinstance(local_llm_instance, LocalLLM):
            raise TypeError("SimpleReasoningAgent requires a LocalLLM instance.")
        self.llm = local_llm_instance
        print("SimpleReasoningAgent: Initialized with LocalLLM instance.")

    def generate(self, prompt: str, max_new_tokens: int = 150) -> str:
        """Generates a response using the LocalLLM."""
        # print(f"SimpleReasoningAgent: Generating response for prompt: {prompt[:100]}...")
        if not self.llm or not self.llm.model:
            return "Error: Reasoning LLM not available or not loaded."
        return self.llm.generate(prompt, max_new_tokens=max_new_tokens)
