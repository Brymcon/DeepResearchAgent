import re

class InputAgent:
    def __init__(self):
        # Order of regex_rules can matter if patterns overlap significantly
        self.regex_rules = {
            # Intent: search_specific_detail (extracts entity and detail)
            r"(?:what is|what's|tell me about|define) (?:the )?(.+?) for (.+)\?": "search_specific_detail",
            r"(?:what is|what's|tell me about|define) (.+)\?": "search_general_what", # Broader what is X?
            r"who is (.+)\?": "search_person",
            r"where is (.+)\?": "search_location",
            r"when did (.+) happen\?": "search_event_time",
            r"(?:how does|how do) (.+) work\?": "explain_process", # Could be reasoning or search
            r"(?:search|find|look up) (?:for )?(.+)": "search_general_command",
            # Add more specific regex rules here
        }
        self.keyword_rules = {
            "search": ["search", "find", "look up", "what is", "who is", "where is", "when is"],
            "reasoning": ["why", "explain", "compare", "discuss", "consider"],
            "task": ["do", "make", "create", "generate", "write", "summarize"],
            "greeting": ["hello", "hi", "hey", "greetings"],
            "farewell": ["bye", "goodbye", "see you", "exit", "quit"]
        }
        self.basic_html_strip_pattern = re.compile(r'<script.*?/script>|<.*?on.*?=.*?>', re.IGNORECASE | re.DOTALL)
        self.all_html_tags_pattern = re.compile(r'<[^>]+>')
        # print("InputAgent: Initialized with regex and keyword rules.")

    def sanitize_input(self, text: str) -> str:
        if not text:
            return ""
        sanitized_text = self.basic_html_strip_pattern.sub('', text)
        sanitized_text = self.all_html_tags_pattern.sub('', sanitized_text)
        sanitized_text = re.sub(r'\s+', ' ', sanitized_text).strip()
        return sanitized_text[:500] # Max length truncation

    def process(self, user_input: str) -> dict:
        sanitized_full_input = self.sanitize_input(user_input)
        processed_content_for_action = sanitized_full_input # Default to full sanitized input
        detected_intent = "general"
        requires_web_search = False
        requires_memory_recall = True # Default to True, most queries will benefit from memory

        # 1. Try Regex Rules First
        for pattern, intent_name in self.regex_rules.items():
            match = re.search(pattern, sanitized_full_input, re.IGNORECASE)
            if match:
                detected_intent = intent_name
                if match.groups():
                    # If regex captures groups, use the last captured group as the primary content for action
                    # This often extracts the core subject of the query.
                    # Example: "what is the capital of France?" -> group(1)='France' if pattern is `capital of (.+)\?`
                    # Example: "what is X for Y?" -> group(1)='X', group(2)='Y'. We might want Y or X or both.
                    # For simplicity, let's use the last non-empty group if multiple exist.
                    extracted = [g for g in match.groups() if g and g.strip()]
                    if extracted:
                        processed_content_for_action = extracted[-1].strip()
                # print(f"InputAgent: Matched regex intent '{detected_intent}', processed_content: '{processed_content_for_action}'")
                break # First regex match wins

        if detected_intent == "general": # If no regex matched, fallback to Keyword Rules
            content_lower = sanitized_full_input.lower()
            for intent_name, keywords in self.keyword_rules.items():
                if any(kw in content_lower for kw in keywords):
                    detected_intent = intent_name
                    # For keyword matches, processed_content_for_action remains the full sanitized_input
                    # print(f"InputAgent: Matched keyword intent '{detected_intent}', processed_content: '{processed_content_for_action}'")
                    break # First keyword category match wins

        # Determine flags based on intent (can be made more sophisticated)
        if detected_intent.startswith("search") or detected_intent == "explain_process":
            requires_web_search = True

        if detected_intent == "greeting" or detected_intent == "farewell":
            requires_memory_recall = False # Greetings/farewells might not need memory

        # print(f"InputAgent: Final - Intent='{detected_intent}', WebSearch={requires_web_search}, MemoryRecall={requires_memory_recall}")
        return {
            "intent": detected_intent,
            "original_input": user_input,
            "sanitized_input": sanitized_full_input,
            "processed_content": processed_content_for_action, # Core topic for search/reasoning
            "requires_web_search": requires_web_search,
            "requires_memory_recall": requires_memory_recall
        }
