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
    This will install all necessary packages, including those for the PDF Vision Assistant (`paddleocr`, `paddlepaddle`, `PyMuPDF`, `numpy`, `opencv-python`).

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

## Website Generator

The `WebsiteGenerator` class transforms a markdown research report into an interactive single-page HTML application. It parses the markdown, generates AI-assisted planning comments for the HTML structure, and then constructs the HTML, CSS, and JavaScript to render the report.

To use it, you can import and call the `generate_website` asynchronous function from the `website_generator` module.

Example usage:
```python
import asyncio
from website_generator import generate_website

async def main():
    sample_markdown_content = """
# My Research Paper

## Table of Contents
- [Section 1](#section-1)
- [Section 2](#section-2)

<a id="section-1"></a>
## Section 1
This is the first section. It has a reference [1].

<a id="section-2"></a>
## Section 2
This is the second section.

# References
1. Example Reference - <https://example.com>
"""
    output_html_file = "my_report.html"

    # Ensure DEEPSEEK_API_KEY and DEEPSEEK_BASE_URL are set as environment variables
    # if you want AI placeholders to be generated for the HTML head comments.
    # If not set, the generator will use default placeholders and log a warning.

    await generate_website(sample_markdown_content, output_html_file)
    print(f"Website generated at {output_html_file}")

if __name__ == "__main__":
    asyncio.run(main())
```
The `generate_website` function requires the markdown content as a string and the desired output path for the HTML file. For the AI-powered generation of planning comments in the HTML head, ensure the `DEEPSEEK_API_KEY` and `DEEPSEEK_BASE_URL` environment variables are set. If these are not available, the generator will fall back to default comments and continue.

## PDF Vision Assistant

The `PDFVisionAssistant` is a tool designed to extract text and analyze layout from PDF files. It utilizes PaddleOCR for optical character recognition, allowing it to process scanned documents or PDFs where text is embedded as images. Furthermore, it uses `PPStructure` (also part of the PaddleOCR ecosystem) to analyze the document layout, identifying elements like titles, text blocks, tables, and figures. This enables a more structured extraction of text content, attempting to preserve a logical reading order.

### Installation

The necessary Python packages for the PDF Vision Assistant are listed in `requirements.txt`. You can install them using:
```bash
pip install -r requirements.txt
```
This command installs `PyMuPDF` (for PDF handling), `paddleocr` (for OCR and structure analysis), `paddlepaddle` (the deep learning framework), `numpy`, and `opencv-python` (for image processing).

**Note:** On its first run for a specific language, PaddleOCR and PPStructure will automatically download the required OCR, detection, and layout analysis models. This means an internet connection will be necessary at that point. Subsequent runs for the same language will use the cached models.

### Usage Examples

#### 1. Basic Text Extraction

This example uses the `extract_text_from_pdf_wrapper` function to get raw text from all pages of a PDF.

```python
from pdf_vision_assistant import extract_text_from_pdf_wrapper
import os

# Create a dummy PDF for testing if you don't have one.
# For a real scenario, you would provide a path to an existing PDF.
try:
    import fitz # PyMuPDF
    if not os.path.exists("sample_basic.pdf"): # Use a different name for clarity
        doc = fitz.open() # New empty PDF
        page = doc.new_page()
        page.insert_text((50, 72), "Hello, this is a test PDF for PaddleOCR.")
        page.insert_text((50, 92), "It contains some sample text on one page.")
        doc.save("sample_basic.pdf")
        doc.close()
        print("Created a dummy 'sample_basic.pdf' for the basic extraction example.")
except ImportError:
    print("PyMuPDF (fitz) is not installed. Cannot create a dummy PDF. Please provide your own PDF.")
except Exception as e:
    print(f"Error creating dummy PDF: {e}. Please provide your own PDF.")


pdf_file_path_basic = "sample_basic.pdf" # Replace with your PDF file path

if os.path.exists(pdf_file_path_basic):
    print(f"\nAttempting to extract basic text from: {pdf_file_path_basic}")
    # Set lang to 'ch' for Chinese, 'en' for English, etc.
    # Check PaddleOCR documentation for supported languages.
    extracted_text = extract_text_from_pdf_wrapper(pdf_file_path_basic, lang='en')

    print("\n--- Extracted Basic Text ---")
    print(extracted_text)
    print("--- End of Basic Text ---")
else:
    print(f"PDF file '{pdf_file_path_basic}' not found. Please create it or provide a valid path.")

```
This wrapper function processes the PDF page by page, converting them to images first, and then uses PaddleOCR to extract text. It's suitable for getting all textual content without detailed structure.

#### 2. Structured Text Extraction (Layout-Aware)

This example uses the `extract_structured_text_from_pdf_wrapper` function, which leverages layout analysis for a more organized text output.

```python
# (Assuming pdf_vision_assistant.py is in the same directory or installed)
from pdf_vision_assistant import extract_structured_text_from_pdf_wrapper
import os

# (You can reuse or adapt the dummy PDF creation code from the previous example if needed)
# Ensure a "sample_structured.pdf" exists or use your own PDF file.
try:
    import fitz
    if not os.path.exists("sample_structured.pdf"): # Use a different name
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((50, 72), "Title of Document") # Mock title
        page.insert_text((50, 102), "This is the first paragraph. It contains some general information and flows across the page.", (50,112))
        page.insert_text((50, 132), "This is a second paragraph, possibly discussing further details. It might be longer.")
        # Add a mock table-like structure
        page.insert_text((50, 162), "Column1 Header | Column2 Header")
        page.insert_text((50, 177), "Row1Data1      | Row1Data2")
        page.insert_text((50, 192), "Row2Data1      | Row2Data2")
        page.insert_text((50, 222), "Another text block after the table content.") # Text after table
        doc.save("sample_structured.pdf")
        doc.close()
        print("Created/updated a dummy 'sample_structured.pdf' for the structured text example.")
except ImportError:
    print("PyMuPDF (fitz) is not installed. Cannot create/update dummy PDF.")
except Exception as e:
    print(f"Error creating/updating dummy PDF: {e}.")


pdf_file_path_structured = "sample_structured.pdf" # Replace with your PDF file path

if os.path.exists(pdf_file_path_structured):
    print(f"\nAttempting to extract structured text from: {pdf_file_path_structured}")
    # This uses the new wrapper that leverages PPStructure for layout analysis
    structured_text = extract_structured_text_from_pdf_wrapper(pdf_file_path_structured, lang='en')

    print("\n--- Extracted Structured Text ---")
    print(structured_text)
    print("--- End of Structured Text ---")
else:
    print(f"PDF file '{pdf_file_path_structured}' not found. Please create it or provide a valid path.")

```
This function first analyzes the PDF's layout using `PPStructure` to identify elements like text blocks, tables, titles, etc. It then reconstructs the text content based on this structure, aiming for a more readable and logically ordered output.
The `use_gpu` parameter can be set to `True` in both wrappers if you have a compatible GPU setup and the GPU-enabled version of PaddlePaddle installed, which can significantly speed up processing.
