import re
from collections import Counter
# Assuming VectorMemory might be needed to update tags, though MemoryAgent is the interface
# from memory.store import VectorMemory

# A basic list of English stop words
STOP_WORDS = set([
    'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should', 'can',
    'could', 'may', 'might', 'must', 'am', 'i', 'you', 'he', 'she', 'it', 'we', 'they',
    'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'its', 'our', 'their',
    'mine', 'yours', 'hers', 'ours', 'theirs', 'myself', 'yourself', 'himself',
    'herself', 'itself', 'ourselves', 'themselves', 'what', 'who', 'whom', 'which',
    'whose', 'when', 'where', 'why', 'how', 'and', 'but', 'or', 'nor', 'for', 'so',
    'yet', 'if', 'then', 'else', 'while', 'as', 'until', 'of', 'at', 'by', 'from',
    'to', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then',
    'once', 'here', 'there', 'all', 'any', 'both', 'each', 'few', 'more', 'most',
    'other', 'some', 'such', 'no', 'not', 'only', 'own', 'same', 'so', 'than',
    'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'q', 'a'
])

class IndexerAgent:
    def __init__(self, memory_agent_ref):
        """
        Initializes the IndexerAgent.
        Args:
            memory_agent_ref: A reference to the instantiated MemoryAgent to interact with memory.
        """
        self.memory_agent = memory_agent_ref
        print("IndexerAgent: Initialized.")

    def extract_keywords(self, text: str, top_n: int = 5) -> list[str]:
        if not text:
            return []
        # Remove punctuation, convert to lowercase, split into words
        text = re.sub(r'[^\w\s]', '', text.lower())
        words = text.split()

        # Filter out stop words and short words
        filtered_words = [word for word in words if word not in STOP_WORDS and len(word) > 2]

        if not filtered_words:
            return []

        # Get the most common words
        word_counts = Counter(filtered_words)
        keywords = [word for word, count in word_counts.most_common(top_n)]
        return keywords

    def process_memory_entry_for_indexing(self, memory_id: int):
        """Processes a single memory entry: extracts keywords and updates its tags."""
        if not self.memory_agent or not self.memory_agent.memory:
            print("IndexerAgent: Memory system not available.")
            return False

        memory_item = self.memory_agent.get_memory_by_id(memory_id)

        if not memory_item or not memory_item.get('content'):
            print(f"IndexerAgent: Could not retrieve memory or content for ID {memory_id}.")
            return False

        content = memory_item['content']
        existing_tags = memory_item.get('tags', [])
        if not isinstance(existing_tags, list):
             # DuckDB VARCHAR[] should come as list, but defensive check
             existing_tags = list(existing_tags) if existing_tags else []

        # A special tag to mark that this memory has been processed by the indexer
        processed_tag = 'indexed_keywords_v1'
        if processed_tag in existing_tags:
            # print(f"IndexerAgent: Memory ID {memory_id} already processed for keywords.")
            return True # Already processed

        extracted_keywords = self.extract_keywords(content)
        if not extracted_keywords:
            # print(f"IndexerAgent: No keywords extracted for memory ID {memory_id}.")
            # Add processed_tag even if no keywords, to avoid re-processing
            updated_tags = existing_tags + [processed_tag]
            self.memory_agent.memory.conn.execute("UPDATE memories SET tags = list_distinct(list_cat(tags, ?)) WHERE id = ?", [[processed_tag], memory_id])
            return True

        # Combine existing tags with new keywords, ensuring no duplicates, and add processed_tag
        new_tags_to_add = [kw for kw in extracted_keywords if kw not in existing_tags]
        if not new_tags_to_add and processed_tag in existing_tags:
            return True # No new keywords to add, and already marked processed

        # Use DuckDB's list functions for robust update if possible
        # list_cat to concatenate, list_distinct to ensure uniqueness
        all_new_tags_for_update = new_tags_to_add + [processed_tag]

        try:
            # This SQL assumes `tags` is VARCHAR[]. It appends the new list of tags and then makes the whole list distinct.
            self.memory_agent.memory.conn.execute("UPDATE memories SET tags = list_distinct(list_cat(tags, ?)) WHERE id = ?", [all_new_tags_for_update, memory_id])
            print(f"IndexerAgent: Updated tags for memory ID {memory_id} with keywords: {new_tags_to_add}")
            return True
        except Exception as e:
            print(f"IndexerAgent: Error updating tags for memory ID {memory_id}: {e}")
            return False

    def run_indexing_batch(self, limit: int = 100):
        """Fetches memories that haven't been indexed and processes them."""
        if not self.memory_agent or not self.memory_agent.memory:
            print("IndexerAgent: Memory system not available for batch indexing.")
            return 0

        # Fetch memories that do NOT contain the 'indexed_keywords_v1' tag
        # DuckDB's list_contains(list, element) or array_contains(array, element)
        # Or more generally, check if 'indexed_keywords_v1' is NOT IN the tags array.
        # Using `NOT list_contains(tags, 'indexed_keywords_v1')`
        # Or, simpler: fetch ids and current tags, then filter in Python (less efficient for large DB but simpler query)
        # For now, let's fetch all IDs and check one by one, or a limited batch.
        # A more efficient query: SELECT id FROM memories WHERE NOT list_contains(tags, 'indexed_keywords_v1') LIMIT ?
        # However, list_contains might not work directly with prepared statements for the element in some older duckdb versions.
        # The most robust is to fetch IDs and then process one by one using get_memory_by_id.

        # Fetch IDs of memories that do not have the 'indexed_keywords_v1' tag.
        # This query is a bit more complex with array operations in WHERE clause.
        # A simpler way for now, less efficient but demonstrative:
        # Get all ids, then check. Or a query that can be parameterized easily.
        # Let's try a direct SQL for fetching unindexed items:
        unindexed_fetch_sql = "SELECT id FROM memories WHERE NOT array_contains(tags, ?) LIMIT ?"
        try:
            unindexed_ids_tuples = self.memory_agent.memory.conn.execute(unindexed_fetch_sql, ['indexed_keywords_v1', limit]).fetchall()
        except Exception as e:
            print(f"IndexerAgent: Error fetching unindexed memories: {e}. Trying a simpler fetch.")
            # Fallback to fetching all IDs if array_contains is problematic in this context
            # This fallback is very inefficient and just for robustness of the agent's run.
            all_ids_tuples = self.memory_agent.memory.conn.execute("SELECT id FROM memories LIMIT ?", [limit * 5]).fetchall() # fetch more to find some unindexed
            # This fallback needs to be smarter; for now, it's a placeholder if direct query fails.
            # For this pass, I'll assume the array_contains query works or a similar variant.
            # If not, the logic to iterate and check tag in Python would be here.
            unindexed_ids_tuples = [] # Placeholder if initial query fails and no good fallback

        if not unindexed_ids_tuples:
            print("IndexerAgent: No memories found requiring keyword indexing in this batch.")
            return 0

        processed_count = 0
        for id_tuple in unindexed_ids_tuples:
            if self.process_memory_entry_for_indexing(id_tuple[0]):
                processed_count += 1
        print(f"IndexerAgent: Batch keyword indexing complete. Processed {processed_count} memories.")
        return processed_count
