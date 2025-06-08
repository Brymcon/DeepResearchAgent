from transformers import AutoTokenizer, AutoModelForCausalLM # Removed pipeline, wasn't used
import torch

class ReasoningAgent:
    def __init__(self, model_name="deepseek-ai/deepseek-coder-1.3b-instruct"):
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"ReasoningAgent: Initializing with model '{model_name}' on device '{self.device}'.")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32, # bfloat16 only on CUDA
                device_map="auto" # Let transformers handle device mapping
            )
            # self.model.to(self.device) # device_map='auto' should handle this
            print(f"ReasoningAgent: Model '{model_name}' loaded successfully on {self.model.device}.")
        except Exception as e:
            print(f"ReasoningAgent: CRITICAL ERROR loading model '{model_name}': {e}")
            # self.model will remain None, handled in generate()

        # System prompt adjusted for CoT, while still aiming for a final usable answer.
        self.system_prompt = (
            "You are a helpful AI assistant with access to memory and web search results. "
            "Please analyze the provided information carefully. "
            "Think step-by-step to arrive at your answer. Explain your reasoning if complex, then provide the final answer."
            # "Respond concisely after your reasoning." # This part can be tricky with CoT.
        )

    def generate(self, query: str, context: str = "") -> str:
        if not self.model or not self.tokenizer:
            print("ReasoningAgent: Model not loaded. Cannot generate response.")
            return "Error: Reasoning model is not available."

        # Constructing the prompt with CoT encouragement
        # Adding a clear instruction for step-by-step thinking.
        prompt_parts = [
            f"### System:\n{self.system_prompt}"
        ]

        if context and context.strip() != "No relevant information found from memory or web search to construct context.":
            prompt_parts.append(f"\n\n### Provided Context (Memory & Search Results):\n{context.strip()}")
        else:
            prompt_parts.append("\n\n### Provided Context:\nNo specific context provided.")

        prompt_parts.append(f"\n\n### User Query:\n{query}")
        prompt_parts.append("\n\n### Instruction:\nFirst, write out your step-by-step reasoning to address the query based on the context. "
                           "After your reasoning, clearly state your final answer to the user's query.")
        prompt_parts.append("\n\n### Assistant Response:")
        # prompt_parts.append("\nReasoning Steps:
1. ...") # Could also prime the model like this

        full_prompt = "".join(prompt_parts)

        # print(f"ReasoningAgent: Generating with prompt:\n{full_prompt}") # For debugging prompt structure

        try:
            inputs = self.tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=4096).to(self.model.device)
            # Ensure max_new_tokens doesn't cause total length to exceed model's absolute max if not handled by truncation alone.
            # Max length for DeepSeek Coder 1.3B is typically 4096 or 16K depending on variant.
            # If inputs.input_ids is already near max_length, max_new_tokens should be smaller.
            available_space_for_new_tokens = self.tokenizer.model_max_length - inputs.input_ids.shape[1]
            if available_space_for_new_tokens <= 0:
                 print("Warning: Prompt is already at max model length. No space for new tokens.")
                 return "Error: The information provided is too long for me to process further and generate a response."

            current_max_new_tokens = min(512, available_space_for_new_tokens - 5) # Reserve a few tokens, ensure it's positive
            if current_max_new_tokens <=0:
                current_max_new_tokens = 50 # a small value if somehow available space is tiny but positive

            outputs = self.model.generate(
                inputs.input_ids,
                max_new_tokens=current_max_new_tokens,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id # Explicitly set eos_token_id for generation
            )
            response = self.tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            ).strip()
            return response
        except Exception as e:
            print(f"ReasoningAgent: Error during model generation: {e}")
            return f"Sorry, I encountered an error while generating a response: {e}"
