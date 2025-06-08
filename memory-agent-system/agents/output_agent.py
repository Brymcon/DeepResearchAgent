import re

class OutputAgent:
    def __init__(self):
        print("OutputAgent: Initialized.")

    def _to_markdown_paragraphs(self, text: str) -> str:
        """Ensures paragraphs are separated by double newlines for basic Markdown feel."""
        if not text or text.strip() == "":
            return "No response content to display."
        # Normalize newlines: replace multiple newlines with a single one first
        text = re.sub(r'\n+', '\n', text).strip()
        # Then, ensure paragraphs (separated by single newlines from LLM) become double for Markdown
        paragraphs = text.split('\n')
        # Filter out any empty paragraphs that might result from split
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        return '\n\n'.join(paragraphs)

    def deliver(self, response_text: str, sources: list = None):
        """
        Processes the raw response text for final delivery to the user (e.g., console).
        Applies basic formatting and will eventually handle source citations.

        Args:
            response_text: The raw text response from the ReasoningAgent.
            sources: Optional list of source information (e.g., dicts with title, link)
                     to be appended to the response. (For future use)
        """
        # print(f"OutputAgent: Received raw response: '{response_text[:100]}...' ") # Debug

        formatted_response = self._to_markdown_paragraphs(response_text)

        # Placeholder for source citations - to be implemented later
        if sources:
            formatted_response += "\n\n---\nSources:\n"
            for i, source in enumerate(sources):
                title = source.get('title', 'Unknown Title')
                link = source.get('link', '#')
                formatted_response += f"[{i+1}] {title} ({link})\n"
            formatted_response = formatted_response.strip()

        # For now, deliver prints to console. This could be extended for other output channels.
        print("\nAssistant:")
        print(formatted_response)

# Example Usage (for testing this agent directly):
# if __name__ == '__main__':
#     output_agent = OutputAgent()
#     raw_text_from_llm = "This is the first paragraph.\nThis is the second paragraph. It might have multiple sentences.\n\nThis could be an intended third paragraph after a double newline from LLM."
#     output_agent.deliver(raw_text_from_llm)

#     raw_text_2 = "Single line response."
#     output_agent.deliver(raw_text_2)

#     raw_text_3 = "Sentence one.\nSentence two.\nSentence three."
#     output_agent.deliver(raw_text_3)

#     mock_sources = [
#         {'title': 'Example Source 1', 'link': 'http://example.com/1'},
#         {'title': 'Another Source', 'link': 'http://example.com/2'}
#     ]
#     output_agent.deliver("Response that uses sources.", sources=mock_sources)
