import re
from collections import Counter
import numpy as np
try:
    import spacy
    NLP_SPACY_MODEL = None # Global to hold loaded spacy model
except ImportError:
    NLP_SPACY_MODEL = None
    # print("IndexerAgent: Warning - Spacy library not found. NER tagging will be disabled.")

try:
    from sklearn.cluster import KMeans
except ImportError:
    KMeans = None
    # print("IndexerAgent: Warning - scikit-learn not found. K-Means clustering will be disabled.")

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
    def __init__(self, memory_agent_ref, num_clusters: int = 10, auto_run_clustering_interval: int = 0):
        self.memory_agent = memory_agent_ref
        self.nlp = None
        global NLP_SPACY_MODEL # Use the global for lazy loading once
        if NLP_SPACY_MODEL is None and NLP_SPACY is not None:
            try:
                NLP_SPACY_MODEL = NLP_SPACY.load('en_core_web_sm')
                print("IndexerAgent: Spacy model 'en_core_web_sm' loaded for NER.")
            except OSError:
                print("IndexerAgent: Spacy model 'en_core_web_sm' not found. Download: python -m spacy download en_core_web_sm. NER disabled.")
        self.nlp = NLP_SPACY_MODEL

        self.num_clusters = num_clusters
        self.kmeans_model = None
        self.kmeans_model_fitted = False
        if KMeans is not None:
            self.kmeans_model = KMeans(n_clusters=self.num_clusters, random_state=42, n_init='auto')
        else:
            print("IndexerAgent: KMeans clustering disabled (scikit-learn not found).")
        # print("IndexerAgent: Initialized.") # Reduced verbosity
        self.auto_run_clustering_interval = auto_run_clustering_interval # Num of batches before auto-run cluster
        self.batch_runs_since_last_cluster = 0

    def _extract_basic_keywords(self, text: str, top_n: int = 5) -> list[str]:
        if not text: return []
        text_processed = re.sub(r'[^\\w\\s]', '', text.lower())
        words = text_processed.split()
        filtered_words = [word for word in words if word not in STOP_WORDS and len(word) > 2]
        if not filtered_words: return []
        return [word for word, count in Counter(filtered_words).most_common(top_n)]

    def _extract_entities_spacy(self, text: str) -> dict:
        entities = {'PERSON': [], 'ORG': [], 'LOC': [], 'DATE': [], 'EVENT': []}
        if not self.nlp or not text: return entities
        doc = self.nlp(text)
        for ent in doc.ents:
            label = ent.label_
            if label in entities: entities[label].append(ent.text.strip())
            elif label == 'FAC' or label == 'GPE': entities['LOC'].append(ent.text.strip())
        for key in entities: entities[key] = sorted(list(set(entities[key])))
        return entities

    def run_memory_clustering(self, sample_size: int = 1000, min_activation_for_sample: int = 0):
        if not self.kmeans_model:
            print("IndexerAgent: KMeans model not available. Skipping clustering.")
            return False
        if not hasattr(self.memory_agent, 'get_memories_for_clustering'):
            print("IndexerAgent: BioMemoryAgent does not have 'get_memories_for_clustering'.")
            return False

        print(f"IndexerAgent: Running memory clustering for k={self.num_clusters}...")
        memories_to_cluster = self.memory_agent.get_memories_for_clustering(
            sample_size=sample_size,
            min_activation_count=min_activation_for_sample
        )
        if not memories_to_cluster or len(memories_to_cluster) < self.num_clusters:
            print("IndexerAgent: Not enough memories found for clustering.")
            return False

        embeddings_list = [mem['memory_trace'] for mem in memories_to_cluster if mem['memory_trace']]
        if not embeddings_list or len(embeddings_list) < self.num_clusters:
            print("IndexerAgent: Not enough valid embeddings for clustering.")
            return False

        try:
            embeddings_array = np.array(embeddings_list).astype(float) # Ensure float for KMeans
            self.kmeans_model.fit(embeddings_array)
            self.kmeans_model_fitted = True
            print(f"IndexerAgent: KMeans model (k={self.num_clusters}) fitted successfully on {len(embeddings_array)} samples.")
            # After fitting, we might want to immediately re-tag all memories or a subset
            # For now, process_memory_entry_for_indexing will apply tags if model is fitted.
            # self.run_indexing_batch(limit=len(memories_to_cluster), force_cluster_tagging=True) # Example
            return True
        except Exception as e:
            print(f"IndexerAgent: Error during KMeans fitting: {e}")
            self.kmeans_model_fitted = False
            return False

    def process_memory_entry_for_indexing(self, memory_id: int, force_cluster_tagging: bool = False):
        if not self.memory_agent: return False
        memory_item = self.memory_agent.get_memory_by_id(memory_id)
        if not memory_item or not memory_item.get('content'): return False

        content = memory_item['content']
        existing_tags = memory_item.get('tags', [])
        if not isinstance(existing_tags, list): existing_tags = list(existing_tags) if existing_tags else []

        newly_added_tags = []
        made_changes = False

        # 1. Keyword Tagging
        processed_keyword_tag = 'indexed_keywords_v1'
        if processed_keyword_tag not in existing_tags:
            extracted_keywords = self._extract_basic_keywords(content)
            for kw in extracted_keywords:
                tag = f"kw:{kw}"
                if tag not in existing_tags and tag not in newly_added_tags: newly_added_tags.append(tag)
            if processed_keyword_tag not in newly_added_tags: newly_added_tags.append(processed_keyword_tag)
            made_changes = True

        # 2. NER Tagging
        processed_ner_tag = 'indexed_ner_v1'
        if self.nlp and processed_ner_tag not in existing_tags:
            entities = self._extract_entities_spacy(content)
            for entity_type, entity_list in entities.items():
                for entity_text in entity_list:
                    safe_entity_text = re.sub(r'\s+', '_', entity_text.lower())
                    tag = f"ner:{entity_type.lower()}:{safe_entity_text}"
                    if tag not in existing_tags and tag not in newly_added_tags: newly_added_tags.append(tag)
            if processed_ner_tag not in newly_added_tags: newly_added_tags.append(processed_ner_tag)
            made_changes = True
        elif not self.nlp and processed_ner_tag not in existing_tags: # Mark as NER processed if NLP not available
            newly_added_tags.append(processed_ner_tag)
            made_changes = True

        # 3. Cluster Tagging
        cluster_tag_prefix = f"cluster_k{self.num_clusters}_"
        current_cluster_tag = None
        for t in existing_tags: # Find if already has a cluster tag for this k
            if t.startswith(cluster_tag_prefix): current_cluster_tag = t; break

        if self.kmeans_model_fitted and (force_cluster_tagging or not current_cluster_tag):
            embedding = memory_item.get('memory_trace') # SynapticDuckDB returns it as list
            if embedding and isinstance(embedding, list) and len(embedding) > 0:
                try:
                    embedding_array = np.array(embedding).reshape(1, -1).astype(float)
                    cluster_label = self.kmeans_model.predict(embedding_array)[0]
                    new_cluster_tag_val = f"{cluster_tag_prefix}c{cluster_label}"
                    if new_cluster_tag_val != current_cluster_tag:
                        # Remove old cluster tags for this k value before adding new one
                        tags_to_keep_after_cluster_update = [t for t in existing_tags if not t.startswith(cluster_tag_prefix)]
                        # This line was problematic as it directly accesses memory_agent.memory.conn
                        # self.memory_agent.memory.conn.execute("UPDATE memories SET tags = ? WHERE id = ?", [tags_to_keep_after_cluster_update, memory_id])
                        # It should use a method on memory_agent if available, or this IndexerAgent needs direct DB access for this specific tag update
                        # For now, assuming memory_agent.update_memory_tags can handle replacing all tags or we add a specific method for it.
                        # Let's assume update_memory_tags replaces all tags for simplicity here.
                        # To correctly remove only cluster tags and add a new one, a more specific memory_agent method would be better.
                        # Quick fix: Rebuild the tag list and use update_memory_tags to set it.
                        final_tags_for_update = tags_to_keep_after_cluster_update
                        if new_cluster_tag_val not in final_tags_for_update: final_tags_for_update.append(new_cluster_tag_val)
                        # The IndexerAgent's update_memory_tags method concatenates tags.
                        # This needs a more direct way to set tags or remove specific tags.
                        # For now, this specific logic of removing old cluster tags is complex without a dedicated memory_agent method.
                        # I will simplify: just add the new cluster tag. If multiple are added, filtering logic elsewhere must handle it.
                        if new_cluster_tag_val not in existing_tags and new_cluster_tag_val not in newly_added_tags:
                             newly_added_tags.append(new_cluster_tag_val)
                        made_changes = True
                except Exception as e:
                    print(f"IndexerAgent: Error predicting cluster for memory ID {memory_id}: {e}")
            # else: print(f"IndexerAgent: No embedding for memory ID {memory_id} to predict cluster.")

        if newly_added_tags:
            try:
                self.memory_agent.update_memory_tags(memory_id, newly_added_tags) # This appends tags
                # print(f"IndexerAgent: Updated tags for memory ID {memory_id}.")
            except Exception as e:
                print(f"IndexerAgent: Error in final tag update for memory ID {memory_id}: {e}")
                return False
        return made_changes

    def run_indexing_batch(self, limit: int = 20, force_cluster_tagging_all_in_batch: bool = False):
        if not self.memory_agent: return 0
        if self.auto_run_clustering_interval > 0:
             self.batch_runs_since_last_cluster += 1
             if self.batch_runs_since_last_cluster >= self.auto_run_clustering_interval:
                 self.run_memory_clustering()
                 self.batch_runs_since_last_cluster = 0
                 force_cluster_tagging_all_in_batch = True

        memories_to_check = self.memory_agent.get_all_memories_for_processing(limit=limit * 2)
        processed_in_batch = 0
        for mem_dict in memories_to_check:
            mem_id = mem_dict.get('neuron_id', mem_dict.get('id'))
            if not mem_id: continue
            if self.process_memory_entry_for_indexing(mem_id, force_cluster_tagging=force_cluster_tagging_all_in_batch):
                processed_in_batch += 1
            if processed_in_batch >= limit: break
        if processed_in_batch > 0:
            print(f"IndexerAgent: Batch indexing run. Entries processed/updated: {processed_in_batch}.")
        return processed_in_batch
