import re
from typing import Tuple, List, Dict, Any # Added type hinting

def _jaccard_similarity(text1: str, text2: str, n_gram_size: int = 1) -> float:
    """Calculates Jaccard similarity between two texts based on word n-grams."""
    if not text1 or not text2:
        return 0.0

    pattern = r'\b\w+\b'
    words1 = re.findall(pattern, text1.lower())
    words2 = re.findall(pattern, text2.lower())

    if not words1 or not words2:
        return 0.0

    def get_ngrams(word_list: List[str], n: int) -> set:
        if n == 1:
            return set(word_list)
        ngrams = set()
        for i in range(len(word_list) - n + 1):
            ngrams.add(tuple(word_list[i:i+n]))
        return ngrams

    ngrams1 = get_ngrams(words1, n_gram_size)
    ngrams2 = get_ngrams(words2, n_gram_size)

    if not ngrams1 or not ngrams2:
        return 0.0

    intersection = len(ngrams1.intersection(ngrams2))
    union = len(ngrams1.union(ngrams2))
    return intersection / union if union > 0 else 0.0


class FusionAgent:
    def fuse(self, user_query: str, memory_results: List[Dict[str, Any]], search_results: List[Dict[str, Any]],
             max_memory_items: int = 3, max_search_items: int = 2,
             similarity_threshold_for_redundancy: float = 0.75, # Increased threshold
             min_memory_strength_to_prioritize: float = 0.7) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Combines knowledge from memory and search results into a structured context string,
        handling basic redundancy by prioritizing strong memories or noting similarity.
        Returns the fused context string and a list of sources actually included.
        """
        context_parts = []
        used_sources = []
        final_memory_items_to_include = []
        final_search_items_to_include = []

        # Limit initial selection
        limited_memory_results = memory_results[:max_memory_items]
        limited_search_results = search_results[:max_search_items]

        # Identify and handle redundancy between memory and search
        search_items_marked_as_redundant_by_memory = set() # Indices of search_results

        for i, mem_item in enumerate(limited_memory_results):
            mem_content = mem_item.get('content', '').strip()
            mem_strength = mem_item.get('synaptic_strength', mem_item.get('importance', 0.0))
            is_strong_memory = mem_strength >= min_memory_strength_to_prioritize
            found_strong_search_overlap = False
            similarity_notes_for_this_mem = []

            for j, search_item in enumerate(limited_search_results):
                if j in search_items_marked_as_redundant_by_memory: continue # Already covered by a stronger memory

                search_snippet = search_item.get('snippet', '').strip()
                if mem_content and search_snippet:
                    sim_score = _jaccard_similarity(mem_content, search_snippet, n_gram_size=2) # Use bigrams
                    if sim_score >= similarity_threshold_for_redundancy:
                        similarity_notes_for_this_mem.append(f"(Note: Similar to Web Result S{j+1}, Jaccard: {sim_score:.2f})")
                        if is_strong_memory:
                            search_items_marked_as_redundant_by_memory.add(j)
                            # print(f"Fusion: Strong Memory M{i+1} overlaps with Search S{j+1}. Prioritizing memory.")
                        found_strong_search_overlap = True # Mark that this memory had an overlap

            final_memory_items_to_include.append({'item': mem_item, 'index': i, 'notes': similarity_notes_for_this_mem})

        # Add search items that were not marked as redundant
        for j, search_item in enumerate(limited_search_results):
            if j not in search_items_marked_as_redundant_by_memory:
                final_search_items_to_include.append({'item': search_item, 'index': j})

        # Construct context string
        if final_memory_items_to_include:
            context_parts.append("## Relevant Information from Memory:")
            for mem_data in final_memory_items_to_include:
                mem_item = mem_data['item']
                original_idx = mem_data['index']
                notes = " " + " ".join(mem_data['notes']) if mem_data['notes'] else ""
                used_sources.append({'type': 'memory', 'id': mem_item.get('neuron_id', mem_item.get('id')), 'data': mem_item})
                context_parts.append(
                    f"### Memory Item M{original_idx+1}{notes}\n"
                    f"   - ID: M{mem_item.get('neuron_id', mem_item.get('id', 'N/A'))}\n"
                    f"   - Source: {mem_item.get('source', 'N/A')}\n"
                    f"   - Bucket: {mem_item.get('bucket_type', 'N/A')}\n"
                    f"   - Strength: {mem_item.get('synaptic_strength', mem_item.get('importance', 0.0)):.2f}\n"
                    f"   - Certainty: {mem_item.get('certainty', 'N/A')}\n"
                    f"   - Content: {mem_item.get('content', '').strip()}"
                )
            context_parts.append("---")
        # else: context_parts.append("No relevant information found in memory for this query."); context_parts.append("---")

        if final_search_items_to_include:
            context_parts.append("## Relevant Information from Web Search:")
            for search_data in final_search_items_to_include:
                search_item = search_data['item']
                original_idx = search_data['index']
                used_sources.append({'type': 'search', 'id': f"S{original_idx+1}", 'data': search_item})
                context_parts.append(
                    f"### Web Result S{original_idx+1}\n"
                    f"   - Title: {search_item.get('title', 'N/A')}\n"
                    f"   - Link: {search_item.get('link', 'N/A')}\n"
                    f"   - Snippet: {search_item.get('snippet', '').strip()}"
                )
            context_parts.append("---")

        if not final_memory_items_to_include and not final_search_items_to_include:
            # This message is if *after filtering* nothing is left.
            # If initially empty, previous fuse() version had messages.
            # For now, let's ensure some message if both are empty.
             if not memory_results and not search_results:
                 return "No information found in memory or from web search.", []
             else:
                 return "After filtering, no information selected for context.", []

        final_context = "\n".join(context_parts)
        return final_context, used_sources
