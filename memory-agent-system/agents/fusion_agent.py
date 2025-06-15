import re

def _jaccard_similarity(text1: str, text2: str) -> float:
    if not text1 or not text2: return 0.0
    pattern = r'\b\w+\b'
    words1 = set(re.findall(pattern, text1.lower()))
    words2 = set(re.findall(pattern, text2.lower()))
    if not words1 or not words2: return 0.0
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    return intersection / union if union > 0 else 0.0

class FusionAgent:
    def fuse(self, user_query: str, memory_results: list, search_results: list,
             max_memory_items: int = 3, max_search_items: int = 2,
             similarity_threshold_for_note: float = 0.7) -> tuple[str, list]:
        """
        Combines knowledge from memory and search results into a structured context string
        and returns the list of sources actually used in the context.
        Args:
            user_query: The original user query.
            memory_results: List of dicts from MemoryAgent.retrieve().
            search_results: List of dicts from SearchAgent.search().
            max_memory_items: Max number of memory items to include.
            max_search_items: Max number of search results to include.
            similarity_threshold_for_note: Threshold for Jaccard similarity to note overlap.
        Returns:
            A tuple: (final_context_string, list_of_used_source_objects)
        """
        context_parts = []
        used_sources = [] # To store sources that contributed to the context

        limited_memory_results = memory_results[:max_memory_items]
        limited_search_results = search_results[:max_search_items]

        if limited_memory_results:
            context_parts.append("## Relevant Information from Memory:")
            for i, mem_item in enumerate(limited_memory_results):
                mem_content = mem_item.get('content', '').strip()
                used_sources.append({'type': 'memory', 'id': mem_item.get('neuron_id', mem_item.get('id')), 'data': mem_item})
                similarity_notes = []
                for j, search_item in enumerate(limited_search_results):
                    search_snippet = search_item.get('snippet', '').strip()
                    if mem_content and search_snippet:
                        sim_score = _jaccard_similarity(mem_content, search_snippet)
                        if sim_score >= similarity_threshold_for_note:
                            similarity_notes.append(f"(Note: Appears similar to Web Result S{j+1}, Jaccard: {sim_score:.2f})")
                note_str = " " + " ".join(similarity_notes) if similarity_notes else ""
                context_parts.append(
                    f"### Memory Item M{i+1}{note_str}\n"
                    f"   - ID: M{mem_item.get('neuron_id', mem_item.get('id', 'N/A'))}\n" # Handle both 'neuron_id' and 'id'
                    f"   - Source: {mem_item.get('source', 'N/A')}\n"
                    f"   - Bucket: {mem_item.get('bucket_type', 'N/A')}\n"
                    f"   - Strength: {mem_item.get('synaptic_strength', mem_item.get('importance', 0.0)):.2f}\n"
                    f"   - Certainty: {mem_item.get('certainty', 'N/A')}\n" # Assuming certainty might be added
                    f"   - Content: {mem_content}"
                )
            context_parts.append("---")
        else:
            context_parts.append("No direct information found in memory for this query.")
            context_parts.append("---")

        if limited_search_results:
            context_parts.append("## Relevant Information from Web Search:")
            for i, search_item in enumerate(limited_search_results):
                used_sources.append({'type': 'search', 'id': f"S{i+1}", 'data': search_item})
                context_parts.append(
                    f"### Web Result S{i+1}\n"
                    f"   - Title: {search_item.get('title', 'N/A')}\n"
                    f"   - Link: {search_item.get('link', 'N/A')}\n"
                    f"   - Snippet: {search_item.get('snippet', '').strip()}"
                )
            context_parts.append("---")

        if not limited_memory_results and not limited_search_results:
            return "No relevant information was found to construct context.", []

        final_context = "\n".join(context_parts)
        return final_context, used_sources
