import unittest
import os
import json
import logging # Added import for logging
from memory import Memory # Assuming memory.py is in the same directory or accessible in PYTHONPATH

class TestMemory(unittest.TestCase):

    def setUp(self):
        """Set up for test methods."""
        self.test_file_path = 'test_agent_memory.json'
        # Ensure no old test file exists before a test
        if os.path.exists(self.test_file_path):
            os.remove(self.test_file_path)
        self.memory_instance = Memory(memory_file_path=self.test_file_path)

    def tearDown(self):
        """Tear down after test methods."""
        if os.path.exists(self.test_file_path):
            os.remove(self.test_file_path)

    def test_initial_load_no_file(self):
        """Test Memory initialization with an empty memory if JSON file doesn't exist."""
        self.assertEqual(self.memory_instance.memory_data, {})
        # Double check no file was created by just initializing
        self.assertFalse(os.path.exists(self.test_file_path))

    def test_add_and_get_memory(self):
        """Test adding items and retrieving them."""
        self.memory_instance.add_to_memory("test_type", "test_key", "test_value")
        self.assertEqual(self.memory_instance.get_from_memory("test_type", "test_key"), "test_value")

        self.memory_instance.add_to_memory("test_type", "test_key_dict", {"a": 1, "b": 2})
        self.assertEqual(self.memory_instance.get_from_memory("test_type", "test_key_dict"), {"a": 1, "b": 2})

    def test_list_memory_type(self):
        """Test listing keys for a memory type."""
        self.memory_instance.add_to_memory("list_type", "key1", "value1")
        self.memory_instance.add_to_memory("list_type", "key2", "value2")
        self.memory_instance.add_to_memory("another_type", "keyA", "valueA")

        list_type_keys = self.memory_instance.list_memory_type("list_type")
        self.assertCountEqual(list_type_keys, ["key1", "key2"]) # Use assertCountEqual for lists where order doesn't matter

        non_existent_keys = self.memory_instance.list_memory_type("non_existent_type")
        self.assertEqual(non_existent_keys, [])

    def test_remove_from_memory(self):
        """Test removing items from memory."""
        self.memory_instance.add_to_memory("remove_type", "key_to_remove", "value_to_remove")
        self.memory_instance.add_to_memory("remove_type", "key_to_keep", "value_to_keep")

        # Test successful removal
        self.assertTrue(self.memory_instance.remove_from_memory("remove_type", "key_to_remove"))
        self.assertIsNone(self.memory_instance.get_from_memory("remove_type", "key_to_remove"))
        self.assertIsNotNone(self.memory_instance.get_from_memory("remove_type", "key_to_keep"))

        # Test removal of non-existent key
        self.assertFalse(self.memory_instance.remove_from_memory("remove_type", "non_existent_key"))

        # Test removal from non-existent type
        self.assertFalse(self.memory_instance.remove_from_memory("non_existent_type", "some_key"))

        # Test that removing the last item in a type removes the type itself
        self.memory_instance.add_to_memory("single_item_type", "single_key", "single_value")
        self.assertTrue(self.memory_instance.remove_from_memory("single_item_type", "single_key"))
        self.assertNotIn("single_item_type", self.memory_instance.memory_data)


    def test_persistence(self):
        """Test that data persists after saving and reloading."""
        self.memory_instance.add_to_memory("persist_type", "persist_key", "persist_value")
        # add_to_memory calls save_memory, so data should be in the file

        # Create a new instance with the same file
        new_memory_instance = Memory(memory_file_path=self.test_file_path)
        self.assertEqual(new_memory_instance.get_from_memory("persist_type", "persist_key"), "persist_value")

    def test_overwrite_existing_key(self):
        """Test that adding an item with an existing type and key overwrites the value."""
        self.memory_instance.add_to_memory("overwrite_type", "overwrite_key", "initial_value")
        self.assertEqual(self.memory_instance.get_from_memory("overwrite_type", "overwrite_key"), "initial_value")

        self.memory_instance.add_to_memory("overwrite_type", "overwrite_key", "new_value")
        self.assertEqual(self.memory_instance.get_from_memory("overwrite_type", "overwrite_key"), "new_value")

    def test_get_non_existent_item(self):
        """Test retrieving non-existent items or types."""
        self.assertIsNone(self.memory_instance.get_from_memory("non_existent_type", "some_key"))

        self.memory_instance.add_to_memory("existing_type", "existing_key", "value")
        self.assertIsNone(self.memory_instance.get_from_memory("existing_type", "non_existent_key"))

    def test_remove_non_existent_item(self):
        """Test that remove_from_memory returns False for non-existent items."""
        self.assertFalse(self.memory_instance.remove_from_memory("some_type", "non_existent_key"))

        self.memory_instance.add_to_memory("another_type_remove", "key1", "val1")
        self.assertFalse(self.memory_instance.remove_from_memory("another_type_remove", "non_existent_key"))
        self.assertFalse(self.memory_instance.remove_from_memory("non_existent_type_remove", "key1"))

    def test_load_invalid_json_file(self):
        """Test loading an improperly formatted JSON file."""
        # This test doesn't need the self.memory_instance from setUp,
        # as it's testing behavior with a pre-existing, corrupted file.
        with open(self.test_file_path, 'w') as f:
            f.write("this is not valid json <<<<<>>>>") # Make it clearly invalid

        # Suppress error logging from the 'memory' module's logger during this specific test
        # The memory.py file uses the root logger implicitly after basicConfig.
        memory_module_logger = logging.getLogger()
        original_logging_level = memory_module_logger.level
        memory_module_logger.setLevel(logging.CRITICAL + 1) # Disable logs below CRITICAL

        try:
            # Create a new Memory instance that will try to load the corrupted file
            corrupted_memory_instance = Memory(memory_file_path=self.test_file_path)
            # It should handle the error and initialize to an empty dictionary
            self.assertEqual(corrupted_memory_instance.memory_data, {})
        finally:
            # Restore logging level
            memory_module_logger.setLevel(original_logging_level)

    def test_save_memory_creates_file(self):
        """Test that save_memory actually creates the file."""
        self.assertFalse(os.path.exists(self.test_file_path))
        self.memory_instance.add_to_memory("save_test", "key", "value") # add_to_memory calls save_memory
        self.assertTrue(os.path.exists(self.test_file_path))
        with open(self.test_file_path, 'r') as f:
            data = json.load(f)
        self.assertEqual(data, {"save_test": {"key": "value"}})

if __name__ == '__main__':
    # Patch the logger in memory.py for the test run to avoid INFO spam during tests
    # This is a bit more direct than trying to capture/suppress logging output streams.
    # We'll set it to WARNING or higher.
    # Note: This assumes memory.py uses a logger named 'logging' or a class logger.
    # If memory.py's logger is named differently, this might need adjustment.
    # For the provided memory.py, it uses `logging.basicConfig` and then `logging.info` etc.
    # So we can adjust the root logger level for the duration of tests.

    # Find the logger used in memory.py. If it's the root logger:
    # test_logger = logging.getLogger() # Get root logger
    # original_level = test_logger.level
    # test_logger.setLevel(logging.WARNING) # Suppress INFO and DEBUG

    # If memory.py had its own logger like: logger = logging.getLogger(__name__)
    # from memory import logger as memory_module_logger # if logger was exposed
    # original_level = memory_module_logger.level
    # memory_module_logger.setLevel(logging.WARNING)


    # Given that memory.py uses `logging.basicConfig` and then `logging.info`,
    # it's simpler to just run tests. The setUp/tearDown handles file creation/deletion.
    # The `test_load_invalid_json_file` specifically handles suppressing error logs for that one test.
    unittest.main()
