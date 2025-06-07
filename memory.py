import json
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Memory:
    """
    A class to manage persistent memory storage in a JSON file.
    """

    def __init__(self, memory_file_path: str = 'agent_memory.json'):
        """
        Initializes the Memory object.

        Args:
            memory_file_path (str): The path to the JSON file used for memory storage.
                                     Defaults to 'agent_memory.json'.
        """
        self.memory_file_path = memory_file_path
        self.memory_data = {}
        self.load_memory()

    def load_memory(self):
        """
        Loads memory data from the JSON file.

        If the file doesn't exist or contains invalid JSON,
        it initializes an empty memory dictionary.
        """
        try:
            if os.path.exists(self.memory_file_path):
                with open(self.memory_file_path, 'r') as f:
                    self.memory_data = json.load(f)
                logging.info(f"Memory loaded successfully from {self.memory_file_path}")
            else:
                self.memory_data = {}
                logging.info(f"No memory file found at {self.memory_file_path}. Initializing empty memory.")
        except FileNotFoundError:
            self.memory_data = {}
            logging.warning(f"Memory file not found at {self.memory_file_path}. Initializing empty memory.")
        except json.JSONDecodeError:
            self.memory_data = {}
            logging.error(f"Error decoding JSON from {self.memory_file_path}. Initializing empty memory.")

    def save_memory(self):
        """
        Saves the current memory data to the JSON file.
        """
        try:
            with open(self.memory_file_path, 'w') as f:
                json.dump(self.memory_data, f, indent=4)
            logging.info(f"Memory saved successfully to {self.memory_file_path}")
        except IOError:
            logging.error(f"Error writing memory to {self.memory_file_path}")

    def add_to_memory(self, memory_type: str, key: str, value: any):
        """
        Adds a key-value pair to the memory under a specific memory type.

        Args:
            memory_type (str): The category or type of the memory item.
            key (str): The key for the memory item.
            value (any): The value of the memory item.
        """
        if memory_type not in self.memory_data:
            self.memory_data[memory_type] = {}
        self.memory_data[memory_type][key] = value
        logging.info(f"Added memory item: Type='{memory_type}', Key='{key}'")
        self.save_memory()

    def get_from_memory(self, memory_type: str, key: str) -> any:
        """
        Retrieves a value from memory given its type and key.

        Args:
            memory_type (str): The category or type of the memory item.
            key (str): The key of the memory item.

        Returns:
            any: The value of the memory item, or None if not found.
        """
        value = self.memory_data.get(memory_type, {}).get(key)
        if value is not None:
            logging.info(f"Retrieved memory item: Type='{memory_type}', Key='{key}'")
        else:
            logging.info(f"Memory item not found: Type='{memory_type}', Key='{key}'")
        return value

    def list_memory_type(self, memory_type: str) -> list[str]:
        """
        Lists all keys stored under a given memory type.

        Args:
            memory_type (str): The category or type of memory.

        Returns:
            list[str]: A list of keys, or an empty list if the type doesn't exist.
        """
        keys = list(self.memory_data.get(memory_type, {}).keys())
        logging.info(f"Listing keys for memory type '{memory_type}': {len(keys)} keys found.")
        return keys

    def remove_from_memory(self, memory_type: str, key: str) -> bool:
        """
        Removes a key-value pair from a specific memory type.

        Args:
            memory_type (str): The category or type of the memory item.
            key (str): The key of the memory item to remove.

        Returns:
            bool: True if successful, False otherwise.
        """
        if memory_type in self.memory_data and key in self.memory_data[memory_type]:
            del self.memory_data[memory_type][key]
            if not self.memory_data[memory_type]: # Remove memory_type if it becomes empty
                del self.memory_data[memory_type]
            logging.info(f"Removed memory item: Type='{memory_type}', Key='{key}'")
            self.save_memory()
            return True
        logging.warning(f"Failed to remove memory item (not found): Type='{memory_type}', Key='{key}'")
        return False

if __name__ == '__main__':
    # Example Usage
    memory_instance = Memory(memory_file_path='test_agent_memory.json')

    # Add items
    memory_instance.add_to_memory("research_reports", "ai_ethics_report", {"content": "Report on AI ethics...", "date": "2023-01-15"})
    memory_instance.add_to_memory("research_reports", "quantum_computing_advances", {"content": "Breakthroughs in quantum...", "date": "2023-02-20"})
    memory_instance.add_to_memory("user_preferences", "theme", "dark")

    # Get items
    report = memory_instance.get_from_memory("research_reports", "ai_ethics_report")
    print(f"AI Ethics Report: {report}")

    theme = memory_instance.get_from_memory("user_preferences", "theme")
    print(f"User Theme: {theme}")

    non_existent = memory_instance.get_from_memory("bookmarks", "google")
    print(f"Non-existent item: {non_existent}")

    # List items
    research_keys = memory_instance.list_memory_type("research_reports")
    print(f"Research Report Keys: {research_keys}")

    empty_keys = memory_instance.list_memory_type("bookmarks")
    print(f"Bookmark Keys: {empty_keys}")

    # Remove items
    memory_instance.remove_from_memory("research_reports", "ai_ethics_report")
    print(f"Research reports after removing one: {memory_instance.list_memory_type('research_reports')}")

    memory_instance.remove_from_memory("user_preferences", "theme")
    print(f"Memory types after removing theme: {memory_instance.list_memory_type('user_preferences')}")
    print(f"All memory data after removals: {memory_instance.memory_data}")

    # Clean up the test file
    if os.path.exists('test_agent_memory.json'):
        os.remove('test_agent_memory.json')
        logging.info("Cleaned up test_agent_memory.json")
