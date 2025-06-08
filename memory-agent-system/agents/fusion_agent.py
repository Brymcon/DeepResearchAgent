class FusionAgent:
    def fuse(self, user_query: str, memory_results: list, search_results: list,
             max_memory_items: int = 3, max_search_items: int = 2) -> str:
        """
        Combines knowledge from memory and search results into a structured context string.

        Args:
            user_query: The original user query.
            memory_results: A list of dictionaries, from MemoryAgent.retrieve().
            search_results: A list of dictionaries, from SearchAgent.search().
            max_memory_items: Max number of memory items to include.
            max_search_items: Max number of search results to include.

        Returns:
            A string containing the structured context for the ReasoningAgent.
        """
        # print(f"FusionAgent: Fusing knowledge for query: '{user_query}'")
        # print(f"FusionAgent: Received {len(memory_results)} memory items, {len(search_results)} search items.")

        context_parts = []

        if memory_results and len(memory_results) > 0:
            context_parts.append("Relevant Information from Memory:")
            # Memory results are assumed to be sorted by relevance already by MemoryAgent
            for i, item in enumerate(memory_results[:max_memory_items]):
                context_parts.append(f"  [Memory Item {i+1}]"
                                   f"    ID: M{item.get('id', 'N/A')}"
                                   f"    Source: {item.get('source', 'N/A')}"
                                   f"    Bucket: {item.get('bucket_type', 'N/A')}"
                                   f"    Importance: {item.get('importance', 0.0):.2f}"
                                   f"    Certainty: {item.get('certainty', 0.0):.2f}"
                                   f"    Content: {item.get('content', '').strip()}")
            context_parts.append("---")
        else:
            context_parts.append("No direct information found in memory.")
            context_parts.append("---")

        if search_results and len(search_results) > 0:
            context_parts.append("Relevant Information from Web Search:")
            # Search results are assumed to be sorted by relevance by the search API
            for i, item in enumerate(search_results[:max_search_items]):
                context_parts.append(f"  [Web Result {i+1}]"
                                   f"    Title: {item.get('title', 'N/A')}"
                                   f"    Link: {item.get('link', 'N/A')}"
                                   f"    Snippet: {item.get('snippet', '').strip()}")
            context_parts.append("---")
        else:
            # Only add this if there were no memory results either, or if search was expected
            # The reasoning agent should know if search was attempted based on context structure
            # context_parts.append("No information found from web search.")
            pass # Avoid being too chatty if search just wasn't triggered

        if not context_parts or (not memory_results and not search_results):
             # This case should be rare now due to default messages above
            return "No relevant information found from memory or web search to construct context."

        final_context = "\n".join(context_parts)
        # print(f"FusionAgent: Final fused context length: {len(final_context)}")
        # print(f"FusionAgent: Fused Context:\n{final_context[:500]}...") # Print snippet of context
        return final_context
