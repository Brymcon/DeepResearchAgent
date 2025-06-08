import re

class InputAgent:
    def __init__(self):
        # Order of regex_rules can matter if patterns overlap significantly
        self.regex_rules = {
            # Example: 'what is the capital of X?' captures X
            r"what is (?:the )?capital of (.+)\?": "search_capital",
            r"who is (.+)\?": "search_person",
            r"(?:search|find|look up) (?:for )?(.+)": "search_general",
            # Add more specific regex rules here
        }
        self.keyword_rules = { # Renamed from routing_rules for clarity
            "search": ["search", "find", "look up", "what is"], # 'what is' is broad, regex might be better
            "reasoning": ["why", "how", "explain", "compare"],
            "task": ["do", "make", "create", "generate"]
        }
        # Basic script and dangerous HTML tag removal pattern
        self.basic_html_strip_pattern = re.compile(r'<script.*?/script>|<.*?on.*?=.*?>', re.IGNORECASE | re.DOTALL)
        # Pattern to remove all other HTML tags
        self.all_html_tags_pattern = re.compile(r'<[^>]+>')
        print("InputAgent: Initialized with regex and keyword rules.")

    def sanitize_input(self, text: str) -> str:
        if not text:
            return ""
        # 1. Strip dangerous tags first
        sanitized_text = self.basic_html_strip_pattern.sub('', text)
        # 2. Strip all other HTML tags
        sanitized_text = self.all_html_tags_pattern.sub('', sanitized_text)
        # 3. Normalize whitespace: collapse multiple spaces/tabs/newlines into single space, then strip ends.
        sanitized_text = re.sub(r'\s+', ' ', sanitized_text).strip()
        # 4. Truncate to max length
        return sanitized_text[:500]

    def process(self, user_input: str) -> dict:
        sanitized_content = self.sanitize_input(user_input)
        # Store original sanitized content for use if no specific entity extracted by regex
        final_content_for_processing = sanitized_content

        # 1. Try Regex Rules First
        for pattern, intent in self.regex_rules.items():
            match = re.search(pattern, sanitized_content, re.IGNORECASE)
            if match:
                # If regex captures a group, use that as the primary content for further processing
                # This helps extract the core subject of the query for some intents
                if match.groups():
                    # Use the first captured group as the key content if available and not empty
                    extracted_content = match.group(1).strip()
                    if extracted_content:
                        final_content_for_processing = extracted_content
                print(f"InputAgent: Matched regex intent '{intent}', content for processing: '{final_content_for_processing}'")
                return {"intent": intent, "original_input": user_input, "sanitized_input": sanitized_content, "processed_content": final_content_for_processing}

        # 2. Fallback to Keyword Rules
        # Using the full sanitized_content for keyword matching if no regex matched
        # (as regex might have narrowed down the content for its own intent processing)
        # However, for keyword matching, usually the broader sanitized_input is better.
        content_lower = sanitized_content.lower()
        for intent, keywords in self.keyword_rules.items():
            if any(kw in content_lower for kw in keywords):
                print(f"InputAgent: Matched keyword intent '{intent}', content for processing: '{sanitized_content}'")
                # For keyword matches, the processed_content is the full sanitized_input
                return {"intent": intent, "original_input": user_input, "sanitized_input": sanitized_content, "processed_content": sanitized_content}

        # 3. Default Intent
        print(f"InputAgent: No specific intent matched. Defaulting to 'general'. Content for processing: '{sanitized_content}'")
        return {"intent": "general", "original_input": user_input, "sanitized_input": sanitized_content, "processed_content": sanitized_content}

# Example Usage:
# if __name__ == '__main__':
#     agent = InputAgent()
#     queries = [
#         "<script>alert('XSS')</script>Search for   the weather in Paris  ",
#         "what is the capital of France?",
#         "who is Albert Einstein?",
#         "find me information about large language models",
#         "Explain quantum physics",
#         "  \n\n  multiple  spaces and lines before this <p>paragraph</p> \n\n",
#         "How does photosynthesis work?"
#     ]
#     for q in queries:
#         result = agent.process(q)
#         print(f"Original: '{q}' -> Processed: {result}\n")
