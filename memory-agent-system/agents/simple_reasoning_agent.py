from .llm_module import LocalLLM
import re
import json # For potentially parsing structured LLM output for sufficiency

class SimpleReasoningAgent:
    def __init__(self, local_llm_instance: LocalLLM):
        if not isinstance(local_llm_instance, LocalLLM):
            raise TypeError("SimpleReasoningAgent requires a LocalLLM instance.")
        self.llm = local_llm_instance
        # print("SimpleReasoningAgent: Initialized.")

    def _check_context_sufficiency(self, query: str, context: str) -> dict:
        if not self.llm or not self.llm.model:
            return {"sufficient": True, "type": "unknown", "details": "LLM not available for sufficiency check.", "confidence": 0.0, "missing_concepts": []}

        sufficiency_prompt = (
            f"You are an AI assistant helping to determine if enough context is available to answer a user's query.
"
            f"User Query: {query}\n"
            f"Provided Context:\n{context}\n\n"
            f"Based ONLY on the Provided Context, evaluate its sufficiency and relevance to directly answer the User Query.
"
            f"Respond in JSON format with the following keys:
"
            f"  \"is_sufficient\": boolean (true if context is sufficient, false otherwise),
"
            f"  \"confidence_score\": float (your confidence from 0.0 to 1.0 that the context is sufficient),
"
            f"  \"missing_information_type\": string (if not sufficient, either 'web_search' or 'more_specific_memories', otherwise 'none'),\n"
            f"  \"missing_details\": string (if not sufficient, brief description of what to search for or what kind of memories are needed, otherwise empty string).
"
            f"Example if NOT sufficient for web search: {{"is_sufficient": false, "confidence_score": 0.3, "missing_information_type": "web_search", "missing_details": "current population of Tokyo"}}\n"
            f"Example if sufficient: {{"is_sufficient": true, "confidence_score": 0.9, "missing_information_type": "none", "missing_details": ""}}\n"
            f"JSON Response:"
        )

        raw_sufficiency_response = self.llm.generate(sufficiency_prompt, max_new_tokens=100)
        # print(f"Sufficiency Check Raw Response: {raw_sufficiency_response}")

        try:
            # Attempt to find JSON within the response (LLMs can be a bit messy)
            json_match = re.search(r'{\s*"is_sufficient".*?}', raw_sufficiency_response, re.DOTALL)
            if json_match:
                parsed_json = json.loads(json_match.group(0))
                is_sufficient = bool(parsed_json.get("is_sufficient", False))
                confidence = float(parsed_json.get("confidence_score", 0.0))
                missing_type = str(parsed_json.get("missing_information_type", "none"))
                missing_details = str(parsed_json.get("missing_details", query if not is_sufficient else ""))

                if not is_sufficient and missing_type not in ["web_search", "more_specific_memories"]:
                    missing_type = "web_search" # Default if type is invalid but says not sufficient
                if not is_sufficient and not missing_details:
                    missing_details = query # Default to original query if details are missing

                return {
                    "sufficient": is_sufficient,
                    "confidence": confidence,
                    "type": missing_type,
                    "details": missing_details
                }
            else:
                # print("Warning: LLM did not return valid JSON for sufficiency. Assuming sufficient.")
                # Fallback parsing based on keywords if JSON fails (less reliable)
                response_lower = raw_sufficiency_response.lower()
                if response_lower.startswith("yes") or "is_sufficient": true" in response_lower : # Simple keyword check
                    return {"sufficient": True, "confidence": 0.6, "type": "sufficient_fallback", "details": raw_sufficiency_response}
                elif "no, need web search for:" in response_lower:
                    details = raw_sufficiency_response[len("No, need web search for:"):].strip()
                    return {"sufficient": False, "confidence": 0.4, "type": "web_search", "details": details if details else query}
                elif "no, need more specific memories about:" in response_lower:
                    details = raw_sufficiency_response[len("No, need more specific memories about:"):].strip()
                    return {"sufficient": False, "confidence": 0.4, "type": "memory_recall", "details": details if details else query}

                return {"sufficient": True, "confidence": 0.1, "type": "unknown_llm_response", "details": raw_sufficiency_response} # Default to sufficient if parsing fails badly

        except json.JSONDecodeError:
            # print(f"Warning: JSONDecodeError parsing sufficiency response. Assuming sufficient. Response: {raw_sufficiency_response}")
            return {"sufficient": True, "confidence": 0.1, "type": "json_decode_error", "details": raw_sufficiency_response}
        except Exception as e:
            # print(f"Error in _check_context_sufficiency parsing: {e}. Assuming sufficient.")
            return {"sufficient": True, "confidence": 0.1, "type": "parsing_error", "details": str(e)}

    def _parse_cot_response(self, llm_full_response: str) -> dict:
        final_answer_marker = "Final Answer:"
        # More robust regex to find marker, allowing for slight variations
        match = re.search(r"(^|\n)Final Answer:(.*)", llm_full_response, re.IGNORECASE | re.DOTALL)

        if match:
            reasoning_steps = llm_full_response[:match.start()].strip()
            final_answer = match.group(2).strip()
            return {"reasoning": reasoning_steps, "answer": final_answer}
        else:
            return {"reasoning": "(Reasoning integrated or not explicitly separated)", "answer": llm_full_response.strip()}

    def generate(self, query: str, context: str, intent: str = "general", attempt_cot: bool = True, check_sufficiency: bool = True) -> dict:
        if not self.llm or not self.llm.model:
            return {"status": "error", "answer": "LLM not available.", "reasoning": "", "sufficiency_data": None}

        sufficiency_data = None
        if check_sufficiency:
            sufficiency_data = self._check_context_sufficiency(query, context)
            if not sufficiency_data["sufficient"]:
                return {
                    "status": "needs_more_data",
                    "type": sufficiency_data["type"],
                    "query_details": sufficiency_data["details"],
                    "answer": "Context is insufficient based on initial check. Further action required.",
                    "reasoning": f"Sufficiency Check: Required {sufficiency_data['type']} for '{sufficiency_data['details']}'. Confidence in current context: {sufficiency_data['confidence']:.2f}",
                    "sufficiency_data": sufficiency_data
                }

        # Dynamic prompt adjustments based on intent
        intent_prompt_injection = ""
        if intent == "explain_process":
            intent_prompt_injection = "Explain this clearly, assuming a beginner's understanding. "
        elif intent == "reasoning":
            intent_prompt_injection = "Provide a detailed step-by-step logical breakdown. "

        prompt = (
            f"Based on the following context, answer the question.\n\n"
            f"Context:\n{context}\n\n"
            f"User Query: {query}\n\n"
            f"{intent_prompt_injection}"
        )
        if attempt_cot:
            prompt += "First, write out your step-by-step reasoning to address the query based on the context. After your reasoning, clearly state the 'Final Answer:'.\nResponse:"
        else:
            prompt += "Response:"

        raw_llm_response = self.llm.generate(prompt, max_new_tokens=300) # Increased max_new_tokens for CoT

        if attempt_cot:
            parsed_response = self._parse_cot_response(raw_llm_response)
            return {
                "status": "success",
                "answer": parsed_response["answer"],
                "reasoning": parsed_response["reasoning"],
                "sufficiency_data": sufficiency_data
            }
        else:
            return {"status": "success", "answer": raw_llm_response, "reasoning": "", "sufficiency_data": sufficiency_data}
