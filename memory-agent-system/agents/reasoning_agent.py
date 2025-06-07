from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch

class ReasoningAgent:
    def __init__(self, model_name="deepseek-ai/deepseek-coder-1.3b-instruct"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.system_prompt = "You are a helpful AI assistant with access to memory. Respond concisely."

    def generate(self, query: str, context: str = ""):
        full_prompt = f"### System:\n{self.system_prompt}\n\n### Memory Context:\n{context}\n\n### User:\n{query}\n\n### Assistant:\n"

        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            inputs.input_ids,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            pad_token_id=self.tokenizer.eos_token_id
        )

        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        return response
