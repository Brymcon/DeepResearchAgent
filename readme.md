# AI-Powered Automated Research & Market Analysis Agent

## Overview

This project implements a sophisticated AI-driven agent capable of performing in-depth automated research on a wide range of topics and conducting focused market research, including SWOT analysis. The agent leverages advanced web searching capabilities, AI-powered content analysis and synthesis, and generates structured, well-referenced reports in both Markdown and PDF formats.

The agent can operate in two primary modes:
1.  **General Research Mode**: Conducts comprehensive investigations into complex topics, exploring various facets, identifying key themes, evaluating evidence, and suggesting avenues for future research. Ideal for producing detailed academic-style reports.
2.  **Market Research Mode**: Focuses on analyzing a specific company and its competitive landscape. It generates sections like Company Profile, Market Overview, Competitor Analysis, and SWOT (Strengths, Weaknesses, Opportunities, Threats) analysis, tailored for market intelligence.

## Key Features

*   **Dual Research Modes**: Easily switch between general in-depth research and targeted market/SWOT analysis.
*   **Comprehensive Web Search**: Utilizes Google Search with multiple strategies (general, news-like, academic-like) to gather diverse and relevant information.
*   **Source Credibility Assessment**: Implements a heuristic-based scoring system to evaluate the credibility of web sources.
*   **AI-Driven Analysis & Synthesis**: Employs large language models (via DeepSeek API by default) to analyze search results, extract key insights, and generate coherent report sections.
*   **Iterative Research Cycles**: Can perform multiple cycles of searching and analysis, using AI-generated follow-up questions to deepen understanding (primarily in General Research Mode).
*   **Structured Report Generation**: Compiles findings into well-organized reports with a table of contents and reference list.
*   **Markdown & PDF Output**: Saves reports in both Markdown (`.md`) for easy editing and PDF (`.pdf`) for professional presentation.
*   **Reference Management**: Automatically tracks and cites sources used in the report.
*   **Customizable Prompts**: Easily modify the AI prompts to tailor the analysis and report generation style.
*   **Environment Variable Configuration**: Securely manage API keys and base URLs through environment variables.
*   **Logging**: Includes structured logging for monitoring and debugging.

## Prerequisites

*   Python 3.8 or higher
*   `pip` (Python package installer)
*   **WeasyPrint Dependencies**: WeasyPrint is used for PDF generation and has system-level dependencies.
    *   **Debian/Ubuntu**: `sudo apt-get install libpango-1.0-0 libcairo2 libgdk-pixbuf2.0-0`
    *   **macOS**: `brew install pango cairo libffi`
    *   **Windows**: Installation can be more complex. Refer to the [WeasyPrint documentation](https://doc.weasyprint.org/stable/install.html#windows) for detailed instructions, often involving GTK3.

## Setup

1.  **Clone the Repository (if applicable)**:
    ```bash
    git clone <your-repository-url>
    cd <repository-directory>
    ```

2.  **Create a Virtual Environment (recommended)**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Python Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set Up Environment Variables**:
    The agent requires API credentials for the language model. Create a `.env` file in the project root or set system environment variables:
    ```env
    DEEPSEEK_API_KEY="your_deepseek_api_key"
    DEEPSEEK_BASE_URL="https://api.deepseek.com" # Or your custom base URL if applicable
    ```
    *Note: You can adapt the script to use other OpenAI-compatible APIs by changing the `OpenAI` client initialization in `features/researchAnalyst.py`.*

## Usage

The main script to run the agent is `features/researchAnalyst.py`.

1.  **Navigate to the features directory (if you are in the project root)**:
    ```bash
    cd features
    ```

2.  **Run the Agent**:
    You can modify the `if __name__ == "__main__":` block in `researchAnalyst.py` to set your desired research topic and mode.

    **Example (General Research Mode):**
    ```python
    # Inside researchAnalyst.py
    if __name__ == "__main__":
        agent = ResearchAgent()
        logger.info("\n--- Starting General Research for In-depth Report --- ")
        general_topic = "The impact of quantum computing on cybersecurity"
        final_report_markdown_general = agent.research_cycle(
            topic=general_topic,
            depth=3, # Number of research cycles
            research_mode="general",
            per_query_delay=5 # Delay in seconds between processing queries in a cycle
        )
        # ... (rest of the example code)
    ```

    **Example (Market Research Mode):**
    ```python
    # Inside researchAnalyst.py
    if __name__ == "__main__":
        agent = ResearchAgent()
        logger.info("\n--- Starting Market Research --- ")
        market_research_topic = "Tesla Inc. and its competitors" # Agent expects "Company Name and its competitors" format
        final_report_markdown_market = agent.research_cycle(
            topic=market_research_topic,
            research_mode="market_research",
            # Depth is typically 1 for the structured queries in market research mode
        )
        # ... (rest of the example code)
    ```

3.  **Execute the script**:
    ```bash
    python researchAnalyst.py
    ```

## Output

The agent will generate:
*   A **Markdown (.md)** file containing the full research report.
*   A **PDF (.pdf)** file, converted from the Markdown content.

These files will be saved in the same directory where the script is run (e.g., the `features` directory by default from the example usage). The filenames will include the research topic and a timestamp.

## Customization

*   **AI Prompts**: The core prompts used to instruct the LLM for analysis and section generation are located in the `generate_report_section_content` method within `researchAnalyst.py`. You can modify these to change the style, focus, or depth of the generated content.
*   **Word Counts**: Target word counts for different report sections are embedded within the prompts and can be adjusted.
*   **API Provider**: While configured for DeepSeek, the `OpenAI` client can be pointed to other compatible API providers by changing the `api_key` and `base_url` in the `__init__` method or via environment variables if you modify the script to use different variable names.
*   **Search Parameters**: Web search parameters like `num_results` and `inter_search_delay` can be adjusted in the `web_search` method or exposed as parameters to the `research_cycle` method.

## Contributing

Contributions are welcome! If you'd like to contribute, please consider:
*   Forking the repository.
*   Creating a new branch for your features or bug fixes.
*   Adding new functionalities, such as support for more search engines or data sources.
*   Improving the credibility assessment algorithm.
*   Enhancing the prompt engineering for better AI outputs.
*   Adding more robust error handling and logging.
*   Writing unit tests.
*   Submitting a pull request with a clear description of your changes.

## License

MIT or Apache 2.0
