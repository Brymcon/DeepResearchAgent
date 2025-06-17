import json
import datetime
from typing import Optional

class InteractionLogger:
    def __init__(self, log_file="interactions.jsonl"):
        self.log_file = log_file

    def log(self, user_input: str, response: str, context: str, reasoning_trace: Optional[str] = None):
        entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "input": user_input,
            "response": response,
            "context": context, # Context can be a string or a structured dict
            "reasoning_trace": reasoning_trace if reasoning_trace else "" # Add reasoning trace
        }
        try:
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(entry) + '\n')
        except Exception as e:
            print(f"Error writing to log file {self.log_file}: {e}")
