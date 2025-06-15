from .llm_module import LocalLLM # Assuming llm_module is in the same directory
import re

class SimpleReasoningAgent:
    def __init__(self, local_llm_instance: LocalLLM):
        if not isinstance(local_llm_instance, LocalLLM):
            raise TypeError("SimpleReasoningAgent requires a LocalLLM instance.")
        self.llm = local_llm_instance
        # print("SimpleReasoningAgent: Initialized.") # Reduced verbosity

    def _check_context_sufficiency(self, query: str, context: str) -> dict:
        """Prompts the LLM to evaluate context sufficiency."""
        if not self.llm or not self.llm.model:
            return {"sufficient": True, "type": "unknown", "details": "LLM not available for sufficiency check."}

        sufficiency_prompt = (
            f"Review the following User Query and the Provided Context.\n"
            f"User Query: {query}\n"
            f"Provided Context:\n{context}\n\n"
            f"Is the Provided Context sufficient and relevant to directly answer the User Query? "
            f"Respond with ONLY one of the following, followed by a brief explanation if not sufficient:
"
            f"1. Yes.\n"
            f"2. No, need web search for: [brief description of what to search for].\n"
            f"3. No, need more specific memories about: [brief description of what kind of memories].\n"
            f"Response:"
        )
        # print(f"Sufficiency Check Prompt: {sufficiency_prompt}")
        raw_sufficiency_response = self.llm.generate(sufficiency_prompt, max_new_tokens=50) # Short response expected
        # print(f"Sufficiency Check Raw Response: {raw_sufficiency_response}")

        sufficiency_response_lower = raw_sufficiency_response.lower()

        if sufficiency_response_lower.startswith("yes"):
            return {"sufficient": True, "type": "sufficient", "details": raw_sufficiency_response}
        elif "no, need web search for:" in sufficiency_response_lower:
            details = raw_sufficiency_response[len("No, need web search for:"):].strip()
            return {"sufficient": False, "type": "web_search", "details": details if details else query}
        elif "no, need more specific memories about:" in sufficiency_response_lower:
            details = raw_sufficiency_response[len("No, need more specific memories about:"):].strip()
            return {"sufficient": False, "type": "memory_recall", "details": details if details else query}
        else: # LLM failed to follow instructions
            # print(f"Warning: LLM failed to follow sufficiency check instructions. Assuming context is sufficient. Response: {raw_sufficiency_response}")
            return {"sufficient": True, "type": "unknown_llm_response", "details": raw_sufficiency_response}

    def _parse_cot_response(self, llm_full_response: str) -> dict:
        """Parses LLM response that might contain CoT reasoning and a final answer."""
        # Simple parsing strategy: Look for a "Final Answer:" type marker.
        # If not found, the whole response is considered the answer.
        # More sophisticated parsing might use XML tags or expect specific section headers.
        final_answer_marker = "Final Answer:"
        marker_pos = llm_full_response.rfind(final_answer_marker) # Find the last occurrence

        if marker_pos != -1:
            reasoning_steps = llm_full_response[:marker_pos].strip()
            final_answer = llm_full_response[marker_pos + len(final_answer_marker):].strip()
            return {"reasoning": reasoning_steps, "answer": final_answer}
        else:
            # If no clear marker, assume the whole thing is the answer, or reasoning is implicit.
            return {"reasoning": "(Reasoning integrated into answer or not explicitly separated)", "answer": llm_full_response}

    def generate(self, query: str, context: str, attempt_cot: bool = True, check_sufficiency: bool = True) -> dict:
        """
        Generates a response, potentially with CoT and context sufficiency check.
        Returns a dictionary with status, answer, and other details.
        """
        if not self.llm or not self.llm.model:
            return {"status": "error", "answer": "LLM not available.", "reasoning": ""}

        # 1. Context Sufficiency Check (Optional)
        if check_sufficiency:
            sufficiency_result = self._check_context_sufficiency(query, context)
            if not sufficiency_result["sufficient"]:
                return {
                    "status": "needs_more_data",
                    "type": sufficiency_result["type"], # 'web_search' or 'memory_recall'
                    "query_details": sufficiency_result["details"], # What to search/recall for
                    "answer": "Context is insufficient. Further action required.",
                    "reasoning": f"Sufficiency Check: {sufficiency_result['details']}"
                }

        # 2. Generate Main Response (Potentially with CoT)
        # The LLM (e.g. LocalLLM using a model from ReasoningAgent in neuroplastic_mind.py)
        # should have its system_prompt updated to encourage CoT if attempt_cot is True.
        # For this SimpleReasoningAgent, we construct the prompt directly.
        # The actual CoT-encouraging prompt is now in LocalLLM or its caller.
        # Here, we construct the prompt based on plan.
        prompt = f"Based on the following context, answer the question.\n\nContext:\n{context}\n\nQuestion: {query}\n"
        if attempt_cot:
            prompt += "First, provide your step-by-step reasoning. After your reasoning, clearly state the 'Final Answer:'.\nAnswer:"
        else:
            prompt += "Answer:"

        raw_llm_response = self.llm.generate(prompt, max_new_tokens=250) # Max tokens for CoT + answer

        if attempt_cot:
            parsed_response = self._parse_cot_response(raw_llm_response)
            return {
                "status": "success",
                "answer": parsed_response["answer"],
                "reasoning": parsed_response["reasoning"]
            }
        else:
            return {"status": "success", "answer": raw_llm_response, "reasoning": ""}
