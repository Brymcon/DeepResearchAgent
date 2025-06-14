from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class LocalLLM:
    def __init__(self, model_name: str = "EleutherAI/gpt-neo-125M", model_path: str = None):
        """
        Initializes the LocalLLM.
        Args:
            model_name: Name of the model to load from Hugging Face (if model_path is None).
            model_path: Optional path to a local model directory or GGUF file.
                        If provided, ctransformers will be used for GGUF.
        """
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_type = "hf" # Hugging Face transformers by default

        load_path = model_path if model_path else model_name
        print(f"LocalLLM: Attempting to initialize model from: {load_path} on device {self.device}")

        try:
            if model_path and model_path.lower().endswith(".gguf"):
                try:
                    from ctransformers import AutoModel as CTAutoModel
                    self.model = CTAutoModel.from_pretrained(model_path, model_type=None) # model_type can often be auto-detected for GGUF
                    self.model_type = "gguf"
                    # GGUF models loaded with ctransformers often don't need a separate tokenizer from HF transformers
                    print(f"LocalLLM: GGUF model loaded successfully from {model_path}.")
                except ImportError:
                    print("LocalLLM: ctransformers library not found. Cannot load GGUF model. Please install with 'pip install ctransformers'.")
                    raise # Re-raise to indicate failure
                except Exception as e_gguf:
                    print(f"LocalLLM: Failed to load GGUF model from {model_path}: {e_gguf}")
                    raise # Re-raise to indicate failure
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name) # model_name for tokenizer
                self.model = AutoModelForCausalLM.from_pretrained(model_name) # model_name for HF model
                self.model.to(self.device)
                self.model_type = "hf"
                print(f"LocalLLM: Hugging Face model '{model_name}' loaded successfully on {self.device}.")

        except Exception as e:
            print(f"LocalLLM: CRITICAL ERROR loading model '{load_path}': {e}")
            print("LocalLLM: Proceeding without a functional model. Generate() will return error messages.")

    def generate(self, prompt: str, max_new_tokens: int = 150) -> str:
        if not self.model:
            return "Error: LLM model is not loaded or failed to initialize."

        # print(f"LocalLLM: Generating response for prompt (first 100 chars): '{prompt[:100]}...' ")
        try:
            if self.model_type == "gguf":
                # ctransformers model call is direct with string input
                # Ensure parameters match ctransformers generate method if different
                return self.model(prompt, max_new_tokens=max_new_tokens, temperature=0.7) # Add other params as needed
            elif self.model_type == "hf" and self.tokenizer:
                inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.tokenizer.model_max_length - max_new_tokens).to(self.device)

                # Ensure max_length for generate doesn't conflict with input token length
                # total_max_length = inputs.input_ids.shape[1] + max_new_tokens
                # if total_max_length > self.tokenizer.model_max_length:
                #    current_max_new_tokens = self.tokenizer.model_max_length - inputs.input_ids.shape[1] - 5 # a little buffer
                # else:
                #    current_max_new_tokens = max_new_tokens

                # if current_max_new_tokens <=0 : current_max_new_tokens = 50 # fallback if prompt is already too long

                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens, # Use original max_new_tokens, truncation handled by tokenizer
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id if self.tokenizer.eos_token_id else 50256 # Common EOS for GPT-2/Neo
                )
                return self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
            else:
                return "Error: Model or tokenizer not properly configured."

        except Exception as e:
            print(f"LocalLLM: Error during text generation: {e}")
            return f"Error during generation: {e}"

# Example Usage (for direct testing of this module)
# if __name__ == '__main__':
#     print("Testing LocalLLM with default small model (EleutherAI/gpt-neo-125M)...")
#     llm = LocalLLM()
#     if llm.model:
#         test_prompt = "Once upon a time,"
#         response = llm.generate(test_prompt, max_new_tokens=50)
#         print(f"Prompt: {test_prompt}")
#         print(f"Response: {response}")
#     else:
#         print("LLM model could not be loaded. Skipping generation test.")

#     print("\nTesting LocalLLM with a placeholder GGUF path...")
#     # To test with a real GGUF, replace 'path/to/your/model.gguf' and ensure ctransformers is installed
#     gguf_llm = LocalLLM(model_path="path/to/your/model.gguf")
#     if gguf_llm.model:
#         test_prompt_gguf = "Explain quantum physics in simple terms:"
#         response_gguf = gguf_llm.generate(test_prompt_gguf, max_new_tokens=100)
#         print(f"Prompt: {test_prompt_gguf}")
#         print(f"Response: {response_gguf}")
#     else:
#         print("GGUF LLM model could not be loaded. Ensure path is correct and ctransformers is installed.")
