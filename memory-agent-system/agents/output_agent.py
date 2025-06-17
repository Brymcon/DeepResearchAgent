import re
from typing import List, Dict, Any, Optional

class OutputAgent:
    def __init__(self):
        pass

    def _to_markdown_paragraphs(self, text: str) -> str:
        if not text or text.strip() == "":
            return "(No response content provided)"
        text = str(text)
        text = re.sub(r'\n{2,}', '\n', text).strip()
        paragraphs = text.split('\n')
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        return '\n\n'.join(paragraphs)

    def deliver(self, response_text: str, sources: Optional[List[Dict[str, Any]]] = None, reasoning_text: Optional[str] = None, show_reasoning: bool = True):
        """
        Processes the raw response text and optional reasoning/sources for final delivery.
        Args:
            response_text: The final answer from the ReasoningAgent.
            sources: Optional list of source objects used for the response.
            reasoning_text: Optional string containing CoT reasoning steps.
            show_reasoning: Boolean flag to control if reasoning text is displayed.
        """

        output_parts = []

        if show_reasoning and reasoning_text and reasoning_text.strip() and reasoning_text != "(Reasoning integrated or not explicitly separated)":
            output_parts.append("**Reasoning Steps:**")
            output_parts.append(self._to_markdown_paragraphs(reasoning_text))
            output_parts.append("\n---\n**Final Answer:**")

        output_parts.append(self._to_markdown_paragraphs(response_text))

        if sources and len(sources) > 0:
            citations = ["\n\n---\n**Sources Considered for Context:**"]
            for i, source_info in enumerate(sources):
                source_type = source_info.get('type')
                source_data = source_info.get('data')
                citation_detail = f"  [{i+1}] "
                if source_type == 'memory' and source_data:
                    mem_id = source_data.get('neuron_id', source_data.get('id', 'N/A'))
                    mem_content_snip = str(source_data.get('content', ''))[:60] + "..."
                    mem_strength = source_data.get('synaptic_strength', source_data.get('importance', None))
                    mem_bucket = source_data.get('bucket_type', None)
                    citation_detail += f"Memory (ID: M{mem_id}"
                    if mem_bucket: citation_detail += f", Bucket: {mem_bucket}"
                    if mem_strength is not None: citation_detail += f", Strength: {mem_strength:.2f}"
                    citation_detail += f"): '{mem_content_snip}'"
                elif source_type == 'search' and source_data:
                    title = source_data.get('title', 'N/A')
                    link = source_data.get('link', '#')
                    citation_detail += f"Web: '{title}' ({link})"
                else:
                    citation_detail += f"Unknown source: {str(source_info)[:100]}"
                citations.append(citation_detail)

            if len(citations) > 1: # Only add if there are actual source strings + header
                 output_parts.append("\n".join(citations))

        print("\nAssistant:")
        print("\n".join(output_parts))
