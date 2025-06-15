import re

class OutputAgent:
    def __init__(self):
        # print("OutputAgent: Initialized.") # Reduced verbosity for library-like class
        pass

    def _to_markdown_paragraphs(self, text: str) -> str:
        if not text or text.strip() == "":
            return "No response content to display."
        text = str(text) # Ensure it's a string
        text = re.sub(r'\n{2,}', '\n', text).strip() # Replace multiple newlines with one
        paragraphs = text.split('\n')
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        return '\n\n'.join(paragraphs)

    def deliver(self, response_text: str, sources: list = None):
        """
        Processes the raw response text for final delivery, applying formatting
        and appending source citations if provided.
        Args:
            response_text: The raw text response from the ReasoningAgent.
            sources: Optional list of source objects (dicts) used for the response.
                     Each dict should have 'type' ('memory' or 'search') and 'data'.
        """
        formatted_response = self._to_markdown_paragraphs(response_text)

        if sources and len(sources) > 0:
            citations = []
            citations.append("\n\n---\n**Sources Considered:**")
            for i, source_info in enumerate(sources):
                source_type = source_info.get('type')
                source_data = source_info.get('data')
                citation = f"  [{i+1}] "
                if source_type == 'memory' and source_data:
                    mem_id = source_data.get('neuron_id', source_data.get('id', 'N/A'))
                    mem_content_snip = str(source_data.get('content', ''))[:70] + "..."
                    citation += f"Memory (ID: M{mem_id}): '{mem_content_snip}'"
                elif source_type == 'search' and source_data:
                    title = source_data.get('title', 'N/A')
                    link = source_data.get('link', '#')
                    citation += f"Web: '{title}' ({link})"
                else:
                    citation += f"Unknown source type or data: {str(source_info)[:100]}"
                citations.append(citation)

            if len(citations) > 1: # Only add if there are actual source strings
                 formatted_response += "\n".join(citations)

        print("\nAssistant:")
        print(formatted_response)
