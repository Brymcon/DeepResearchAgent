import json
import datetime

class InteractionLogger:
    def __init__(self, log_file="interactions.jsonl"):
        self.log_file = log_file
        # Attempt to create the directory for the log file if it doesn't exist
        # This is a common pattern but might be better handled by the calling code
        # or by ensuring the main script runs from a location where it has write permissions.
        # For now, assuming the directory of main.py is writable.

    def log(self, user_input, response, context):
        entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "input": user_input,
            "response": response,
            "context": context # Context can be a string or a structured dict
        }
        try:
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(entry) + '\n')
        except Exception as e:
            print(f"Error writing to log file {self.log_file}: {e}")
