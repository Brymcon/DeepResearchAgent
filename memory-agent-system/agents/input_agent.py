class InputAgent:
    def __init__(self):
        self.routing_rules = {
            "search": ["search", "find", "look up", "what is"],
            "reasoning": ["why", "how", "explain", "compare"],
            "task": ["do", "make", "create", "generate"]
        }

    def process(self, user_input: str):
        # Basic sanitization
        sanitized = user_input.strip()[:500]

        # Route to appropriate handler
        for intent, keywords in self.routing_rules.items():
            if any(kw in sanitized.lower() for kw in keywords):
                return {"intent": intent, "content": sanitized}

        return {"intent": "general", "content": sanitized}
