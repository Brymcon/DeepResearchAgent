import re
from collections import Counter
# from agents.bio_memory_agent import BioMemoryAgent # Type hinting if needed

try:
    import spacy
    # Load a small English model. User needs to download this: python -m spacy download en_core_web_sm
    # To avoid error on startup if model not downloaded, load it lazily or wrap in try-except here.
    NLP_SPACY = None
except ImportError:
    NLP_SPACY = None
    print("IndexerAgent: Warning - Spacy library not found. NER tagging will be disabled. Try 'pip install spacy' and download a model.")

STOP_WORDS = set([
    'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should', 'can',
    # ... (keeping existing stop words for brevity, assume they are complete enough)
    'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'q', 'a'
])

class IndexerAgent:
    def __init__(self, memory_agent_ref):
        self.memory_agent = memory_agent_ref
        self.nlp = None
        if NLP_SPACY:
            try:
                self.nlp = NLP_SPACY.load('en_core_web_sm')
                print("IndexerAgent: Spacy model 'en_core_web_sm' loaded for NER.")
            except OSError:
                print("IndexerAgent: Spacy model 'en_core_web_sm' not found. Download it: python -m spacy download en_core_web_sm")
                print("IndexerAgent: NER tagging will be disabled.")
                self.nlp = None # Ensure it's None if model loading fails
        # print("IndexerAgent: Initialized.") # Reduced verbosity

    def _extract_basic_keywords(self, text: str, top_n: int = 5) -> list[str]:
        if not text: return []
        text_processed = re.sub(r'[^\w\s]', '', text.lower())
        words = text_processed.split()
        filtered_words = [word for word in words if word not in STOP_WORDS and len(word) > 2]
        if not filtered_words: return []
        return [word for word, count in Counter(filtered_words).most_common(top_n)]

    def _extract_entities_spacy(self, text: str) -> dict:
        entities = {'PERSON': [], 'ORG': [], 'LOC': [], 'DATE': [], 'EVENT': []} # Include EVENT
        if not self.nlp or not text:
            return entities
        doc = self.nlp(text)
        for ent in doc.ents:
            label = ent.label_
            if label in entities:
                entities[label].append(ent.text.strip())
            elif label == 'FAC' or label == 'GPE': # Facility, Geo-Political Entity often map to Location
                 entities['LOC'].append(ent.text.strip())
        # Deduplicate entities within each category
        for key in entities:
            entities[key] = sorted(list(set(entities[key])))
        return entities

    def process_memory_entry_for_indexing(self, memory_id: int):
        if not self.memory_agent or not hasattr(self.memory_agent, 'get_memory_by_id') or not hasattr(self.memory_agent, 'update_memory_tags'):
            # print("IndexerAgent: MemoryAgent not properly configured for indexing.")
            return False

        memory_item = self.memory_agent.get_memory_by_id(memory_id)
        if not memory_item or not memory_item.get('content'):
            return False

        content = memory_item['content']
        existing_tags = memory_item.get('tags', [])
        if not isinstance(existing_tags, list): existing_tags = list(existing_tags) if existing_tags else []

        processed_keyword_tag = 'indexed_keywords_v1'
        processed_ner_tag = 'indexed_ner_v1'
        already_keyword_indexed = processed_keyword_tag in existing_tags
        already_ner_indexed = processed_ner_tag in existing_tags or not self.nlp # If no spacy, consider NER done

        if already_keyword_indexed and already_ner_indexed:
            return True # Nothing to do

        newly_added_tags = []

        # 1. Basic Keyword Extraction (if not already done)
        if not already_keyword_indexed:
            extracted_keywords = self._extract_basic_keywords(content)
            for kw in extracted_keywords:
                tag = f"kw:{kw}" # Add prefix for keyword tags
                if tag not in existing_tags and tag not in newly_added_tags:
                    newly_added_tags.append(tag)
            if processed_keyword_tag not in existing_tags and processed_keyword_tag not in newly_added_tags:
                 newly_added_tags.append(processed_keyword_tag)

        # 2. NER Tagging (if not already done and spacy is available)
        if not already_ner_indexed and self.nlp:
            entities = self._extract_entities_spacy(content)
            for entity_type, entity_list in entities.items():
                for entity_text in entity_list:
                    # Sanitize entity_text for use in a tag (e.g., replace spaces, lowercase)
                    safe_entity_text = re.sub(r'\s+', '_', entity_text.lower())
                    tag = f"ner:{entity_type.lower()}:{safe_entity_text}"
                    if tag not in existing_tags and tag not in newly_added_tags:
                        newly_added_tags.append(tag)
            if processed_ner_tag not in existing_tags and processed_ner_tag not in newly_added_tags:
                newly_added_tags.append(processed_ner_tag)

        if newly_added_tags:
            try:
                self.memory_agent.update_memory_tags(memory_id, newly_added_tags)
                # print(f"IndexerAgent: Updated tags for memory ID {memory_id} with: {newly_added_tags}")
                return True
            except Exception as e:
                print(f"IndexerAgent: Error updating tags for memory ID {memory_id}: {e}")
                return False
        elif not already_keyword_indexed or not already_ner_indexed: # Ensure processed tags are added even if no new content tags
            # This case handles adding only the 'processed' tags if no new content tags were found
            ensure_processed_tags = []
            if not already_keyword_indexed and processed_keyword_tag not in existing_tags: ensure_processed_tags.append(processed_keyword_tag)
            if not already_ner_indexed and processed_ner_tag not in existing_tags: ensure_processed_tags.append(processed_ner_tag)
            if ensure_processed_tags:
                 self.memory_agent.update_memory_tags(memory_id, ensure_processed_tags)
            return True
        return True # No new tags to add, already marked or nothing to do

    def run_indexing_batch(self, limit: int = 20): # Reduced default limit for batch
        if not self.memory_agent or not hasattr(self.memory_agent, 'get_all_memories_for_processing'):
            # print("IndexerAgent: MemoryAgent not properly configured for batch indexing.")
            return 0

        # Fetch memories that do NOT contain EITHER 'indexed_keywords_v1' OR 'indexed_ner_v1' (if spacy available)
        # This requires fetching tags. For simplicity, get_all_memories_for_processing can fetch ids and tags.
        # Or, we process a fixed number of oldest memories not having *both* tags.
        # For now, let's fetch a batch and process_memory_entry_for_indexing will handle individual checks.
        # This is less efficient than a targeted SQL query but simpler to implement agent-side.
        memories_to_check = self.memory_agent.get_all_memories_for_processing(limit=limit * 2) # Fetch more to find candidates

        processed_in_batch = 0
        for mem_dict in memories_to_check:
            mem_id = mem_dict.get('neuron_id', mem_dict.get('id'))
            if not mem_id: continue

            # Check if already processed (both keyword and NER if applicable)
            current_tags = mem_dict.get('tags', [])
            if not isinstance(current_tags, list): current_tags = list(current_tags) if current_tags else []

            keyword_done = 'indexed_keywords_v1' in current_tags
            ner_done = 'indexed_ner_v1' in current_tags or not self.nlp

            if keyword_done and ner_done:
                continue

            if self.process_memory_entry_for_indexing(mem_id):
                processed_in_batch += 1
                if processed_in_batch >= limit:
                    break

        if processed_in_batch > 0:
            print(f"IndexerAgent: Batch indexing processed {processed_in_batch} memories.")
        # else:
            # print("IndexerAgent: No memories required indexing in this batch.")
        return processed_in_batch
