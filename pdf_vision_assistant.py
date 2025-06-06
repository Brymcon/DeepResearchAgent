import os
import fitz  # PyMuPDF
from paddleocr import PaddleOCR, PPStructure
import logging
import numpy as np
import cv2

# Configure logger
logger = logging.getLogger(__name__)

class PDFVisionAssistant:
    """
    An agent that extracts text from PDF files using PaddleOCR.
    It can also attempt to understand document structure.
    """
    def __init__(self, lang='en', use_gpu=False, **kwargs):
        """
        Initializes the PDFVisionAssistant with PaddleOCR.
        Args:
            lang (str): Language code for OCR (e.g., 'en', 'ch').
            use_gpu (bool): Whether to use GPU for OCR.
            **kwargs: Additional keyword arguments for PaddleOCR initialization.
        """
        logger.info(f"Initializing PaddleOCR with lang='{lang}', use_gpu={use_gpu}")
        try:
            # Initialize PaddleOCR. You might need to adjust model paths or other parameters.
            # show_log=False can be used to reduce console output from PaddleOCR
            self.ocr = PaddleOCR(use_angle_cls=True, lang=lang, use_gpu=use_gpu, show_log=False, **kwargs)
            logger.info("PaddleOCR initialized successfully.")
            # For layout analysis, table recognition, etc.
            logger.info(f"Initializing PPStructure with lang='{lang}', use_gpu={use_gpu}")
            self.structure_analyzer = PPStructure(lang=lang, use_gpu=use_gpu, show_log=False, **kwargs)
            logger.info("PPStructure initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize PaddleOCR or PPStructure: {e}", exc_info=True)
            # Potentially re-raise or handle to prevent agent from being used if OCR engine fails
            raise

    def _pdf_page_to_image(self, pdf_document, page_num, zoom=2):
        """
        Converts a single PDF page to a PNG image.
        Args:
            pdf_document (fitz.Document): The opened PDF document.
            page_num (int): The page number to convert.
            zoom (int): Zoom factor for rendering the image (higher zoom = higher resolution).
        Returns:
            bytes: The image data as bytes, or None if conversion fails.
        """
        try:
            page = pdf_document.load_page(page_num)
            mat = fitz.Matrix(zoom, zoom) # zoom factor for x and y
            pix = page.get_pixmap(matrix=mat)
            img_bytes = pix.tobytes("png")
            logger.debug(f"Successfully converted page {page_num} to image.")
            return img_bytes
        except Exception as e:
            logger.error(f"Error converting PDF page {page_num} to image: {e}", exc_info=True)
            return None

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extracts text from all pages of a given PDF file.
        Args:
            pdf_path (str): Path to the PDF file.
        Returns:
            str: Concatenated text extracted from all pages.
                 Returns an error message string if processing fails.
        """
        if not os.path.exists(pdf_path):
            logger.error(f"PDF file not found: {pdf_path}")
            return "Error: PDF file not found."

        extracted_text_parts = []
        logger.info(f"Opening PDF: {pdf_path}")
        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            logger.error(f"Failed to open PDF {pdf_path}: {e}", exc_info=True)
            return f"Error: Could not open PDF file {pdf_path}."

        logger.info(f"Processing {len(doc)} pages from {pdf_path}...")
        for i, page_num in enumerate(range(len(doc))):
            logger.info(f"Processing page {page_num + 1}/{len(doc)}...")
            img_bytes = self._pdf_page_to_image(doc, page_num)
            if img_bytes:
                try:
                    # The result is a list of lists, where each inner list contains
                    # [bounding_box, (text, confidence_score)]
                    result = self.ocr.ocr(img_bytes, cls=True)
                    if result and result[0] is not None: # Ensure result is not None and has content
                        page_text_parts = []
                        for line in result[0]: # Iterate through lines found on the page
                            if line and len(line) == 2 and isinstance(line[1], tuple) and len(line[1]) == 2:
                                text_content = line[1][0]
                                confidence = line[1][1]
                                logger.debug(f"Extracted line: '{text_content}' (Confidence: {confidence:.4f})")
                                page_text_parts.append(text_content)
                            else:
                                logger.warning(f"Unexpected line format in OCR result for page {page_num + 1}: {line}")

                        extracted_text_parts.append("\n".join(page_text_parts))
                        logger.info(f"Successfully extracted text from page {page_num + 1}.")
                    else:
                        logger.info(f"No text found or empty result for page {page_num + 1}.")
                        extracted_text_parts.append(f"[Page {page_num + 1}: No text detected]")

                except Exception as e:
                    logger.error(f"Error during OCR processing for page {page_num + 1} of {pdf_path}: {e}", exc_info=True)
                    extracted_text_parts.append(f"[Page {page_num + 1}: OCR error]")
            else:
                logger.warning(f"Skipping page {page_num + 1} due to image conversion error.")
                extracted_text_parts.append(f"[Page {page_num + 1}: Image conversion error]")

        logger.info(f"Finished processing PDF: {pdf_path}")
        doc.close()
        return "\n\n---\nPage Break\n---\n\n".join(extracted_text_parts)

    # TODO: Add method for structured data extraction if PPStructure is to be used.
    # def extract_structure_from_pdf(self, pdf_path: str):
    #     pass

    def analyze_pdf_layout_and_text(self, pdf_path: str) -> list:
        """
        Analyzes a PDF to extract text and layout information using PPStructure.
        Args:
            pdf_path (str): Path to the PDF file.
        Returns:
            list: A list of page results. Each item in the list corresponds to a page
                  and contains a dictionary with 'page_number' and 'layout_elements'.
                  'layout_elements' is a list of dictionaries, where each dictionary
                  represents a detected element (e.g., text block, table, figure)
                  and includes its 'type', 'bbox' (bounding box), and 'text' (if applicable).
                  Returns an error message string if processing fails.
        """
        if not os.path.exists(pdf_path):
            logger.error(f"PDF file not found: {pdf_path}")
            return [{'error': "PDF file not found."}]

        logger.info(f"Opening PDF for layout analysis: {pdf_path}")
        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            logger.error(f"Failed to open PDF {pdf_path}: {e}", exc_info=True)
            return [{'error': f"Could not open PDF file {pdf_path}."}]

        all_page_results = []
        logger.info(f"Analyzing layout for {len(doc)} pages from {pdf_path}...")

        for page_num in range(len(doc)):
            page_data = {'page_number': page_num + 1, 'layout_elements': []}
            logger.info(f"Analyzing page {page_num + 1}/{len(doc)}...")

            # Convert page to image (in-memory)
            # PPStructure expects an image path or an OpenCV ndarray.
            # To avoid temp files, we'll pass an ndarray.
            # This requires converting fitz pixmap to an OpenCV image.
            img_bytes = self._pdf_page_to_image(doc, page_num, zoom=3) # Higher zoom for better structure analysis
            if not img_bytes:
                logger.warning(f"Skipping page {page_num + 1} due to image conversion error.")
                page_data['layout_elements'].append({'type': 'error', 'text': 'Image conversion error'})
                all_page_results.append(page_data)
                continue

            try:
                # Convert image bytes to OpenCV ndarray
                # NumPy and cv2 are imported at the top of the file.
                nparr = np.frombuffer(img_bytes, np.uint8)
                img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if img_np is None:
                    logger.error(f"Failed to decode image for page {page_num + 1}.")
                    page_data['layout_elements'].append({'type': 'error', 'text': 'Image decoding error'})
                    all_page_results.append(page_data)
                    continue

                # Use PPStructure for layout analysis
                # The result from PPStructure is a list of dicts, each describing an element.
                # Each dict has 'type', 'bbox', 'res' (which contains OCR result for text elements)
                logger.debug(f"Sending page {page_num+1} image to PPStructure analyzer.")
                # PPStructure's `__call__` method takes the image (ndarray or path)
                structure_result = self.structure_analyzer(img_np)
                logger.debug(f"Received {len(structure_result)} elements from PPStructure for page {page_num+1}.")

                for element in structure_result:
                    element_type = element.get('type', 'unknown')
                    bbox = element.get('bbox', []) # [x_min, y_min, x_max, y_max]

                    text_content = ""
                    if 'res' in element and element['res'] is not None:
                        # 'res' contains a list of detected text lines, similar to PaddleOCR's output
                        # Each item: [[box_points], ("text", confidence)]
                        lines = element['res']
                        text_lines = []
                        for line_info in lines:
                            if isinstance(line_info, dict) and 'text' in line_info : # Newer PPStructure format
                                text_lines.append(line_info['text'])
                            elif isinstance(line_info, (list, tuple)) and len(line_info) == 2 and isinstance(line_info[1], (list,tuple)) and len(line_info[1]) == 2: # Older format
                                text_lines.append(line_info[1][0])
                        text_content = "\n".join(text_lines)

                    page_data['layout_elements'].append({
                        'type': element_type,
                        'bbox': bbox,
                        'text': text_content
                    })
                    logger.debug(f"Page {page_num+1}: Found element type '{element_type}' with text length {len(text_content)} at {bbox}")

            except ImportError:
                logger.error("NumPy or OpenCV (cv2) is not installed. These are required for passing images to PPStructure in memory.", exc_info=True)
                # If critical, this error should probably be raised or handled more globally
                page_data['layout_elements'].append({'type': 'error', 'text': 'NumPy or OpenCV not installed.'})
                # Stop further processing if dependencies are missing for all pages
                all_page_results.append(page_data) # Add current page error
                # Add error message for subsequent pages to avoid repeated attempts
                for remaining_page_num in range(page_num + 1, len(doc)):
                    all_page_results.append({
                        'page_number': remaining_page_num + 1,
                        'layout_elements': [{'type': 'error', 'text': 'NumPy or OpenCV not installed. Aborting further layout analysis.'}]
                    })
                doc.close()
                return all_page_results


            except Exception as e:
                logger.error(f"Error during PPStructure analysis for page {page_num + 1}: {e}", exc_info=True)
                page_data['layout_elements'].append({'type': 'error', 'text': f'PPStructure analysis error: {str(e)}'})

            all_page_results.append(page_data)

        logger.info(f"Finished layout analysis for PDF: {pdf_path}")
        doc.close()
        return all_page_results

    def generate_structured_text_from_layout(self, layout_results: list) -> str:
        """
        Generates a single string of text, structured based on layout information.
        It attempts to reconstruct paragraphs and maintain a reading order.

        Args:
            layout_results (list): The output from `analyze_pdf_layout_and_text`.
                                   A list of page dictionaries.
        Returns:
            str: A single string containing the structured text from the entire document.
        """
        full_document_text_parts = []
        logger.info("Generating structured text from layout results...")

        if not layout_results or (len(layout_results) == 1 and layout_results[0].get('error')):
            logger.warning("Layout results are empty or contain only an error. Cannot generate structured text.")
            if layout_results and layout_results[0].get('error'):
                return f"Error in layout analysis: {layout_results[0]['error']}"
            return "Error: No layout information available to generate structured text."

        for page_result in layout_results:
            page_number = page_result.get('page_number', 'Unknown')
            logger.info(f"Processing structured text for page {page_number}...")

            elements = page_result.get('layout_elements', [])
            if not elements or (len(elements) == 1 and elements[0].get('type') == 'error'):
                error_text = elements[0].get('text', 'Unknown error') if elements else 'No elements found'
                logger.warning(f"Page {page_number} has errors or no elements: {error_text}")
                full_document_text_parts.append(f"[Page {page_number}: {error_text}]")
                continue

            page_text_blocks = []

            # Sort elements by their vertical position (top of bounding box), then horizontal.
            # This helps in establishing a general reading order.
            # Bbox is [x_min, y_min, x_max, y_max]
            try:
                elements.sort(key=lambda el: (el.get('bbox', [0,0,0,0])[1], el.get('bbox', [0,0,0,0])[0]))
            except Exception as e:
                logger.warning(f"Could not sort elements for page {page_number} due to missing or invalid bbox: {e}. Processing in original order.")

            for element in elements:
                element_type = element.get('type', 'unknown')
                text = element.get('text', '').strip()
                bbox = element.get('bbox', [])

                if not text: # Skip elements with no text
                    continue

                # Basic structuring based on type (can be expanded)
                if element_type in ['title', 'header', 'footer']:
                    page_text_blocks.append(f"\n## {text} ##\n") # Markdown-like emphasis for titles/headers
                elif element_type == 'table':
                    # For tables, PPStructure might return complex nested structures or just lines of text.
                    # Here, we're just appending the raw text extracted from table cells.
                    # A more sophisticated approach would format it as a text-based table.
                    page_text_blocks.append(f"\n[Table Data]\n{text}\n[End Table Data]\n")
                elif element_type == 'figure':
                     page_text_blocks.append(f"\n[Figure: {text}]\n") # Text is often the caption
                elif element_type == 'text': # Generic text block
                    page_text_blocks.append(text)
                else: # Other types or unknown
                    page_text_blocks.append(text)

            # Join text blocks for the current page with double newlines to simulate paragraphs/sections
            full_document_text_parts.append("\n\n".join(page_text_blocks))

        # Join text from all pages, separated by a clear page break marker
        return "\n\n--- Page Break ---\n\n".join(full_document_text_parts)


def extract_text_from_pdf_wrapper(pdf_path: str, lang: str = 'en', use_gpu: bool = False) -> str:
    """
    A wrapper function to easily extract text from a PDF using PDFVisionAssistant.

    Args:
        pdf_path (str): Path to the PDF file.
        lang (str): Language code for OCR (e.g., 'en', 'ch'). Defaults to 'en'.
        use_gpu (bool): Whether to attempt using GPU for PaddleOCR. Defaults to False.

    Returns:
        str: Extracted text from the PDF, or an error message if extraction fails.
    """
    try:
        # Configure basic logging if no handlers are configured yet
        # This helps ensure that users see logs if they run this script directly
        # or if the main application hasn't configured logging yet.
        if not logging.getLogger().hasHandlers():
            logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        main_logger = logging.getLogger(__name__) # Use the module's logger
        main_logger.info(f"PDFVisionAssistant wrapper called for PDF: {pdf_path} with lang='{lang}', use_gpu={use_gpu}")

        assistant = PDFVisionAssistant(lang=lang, use_gpu=use_gpu)
        text_content = assistant.extract_text_from_pdf(pdf_path)

        main_logger.info(f"Successfully processed PDF: {pdf_path} using wrapper.")
        return text_content
    except Exception as e:
        # Use a logger that's likely to be configured (e.g., the module's logger)
        logging.getLogger(__name__).error(f"Error in PDFVisionAssistant wrapper for PDF {pdf_path}: {e}", exc_info=True)
        return f"Error: An unexpected error occurred in the PDF processing wrapper: {str(e)}"


def extract_structured_text_from_pdf_wrapper(pdf_path: str, lang: str = 'en', use_gpu: bool = False) -> str:
    """
    A wrapper function to extract text from a PDF, structured using layout analysis.

    Args:
        pdf_path (str): Path to the PDF file.
        lang (str): Language code for OCR (e.g., 'en', 'ch'). Defaults to 'en'.
        use_gpu (bool): Whether to attempt using GPU for PaddleOCR. Defaults to False.

    Returns:
        str: Extracted text from the PDF, structured based on layout,
             or an error message if extraction fails.
    """
    try:
        if not logging.getLogger().hasHandlers():
            logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        main_logger = logging.getLogger(__name__)
        main_logger.info(f"Structured PDF extraction wrapper called for PDF: {pdf_path} with lang='{lang}', use_gpu={use_gpu}")

        assistant = PDFVisionAssistant(lang=lang, use_gpu=use_gpu)

        layout_results = assistant.analyze_pdf_layout_and_text(pdf_path)

        # Check if layout_results indicate an error (e.g., list with a single dict having an 'error' key)
        if isinstance(layout_results, list) and len(layout_results) > 0 and isinstance(layout_results[0], dict) and 'error' in layout_results[0]:
            error_msg = layout_results[0]['error']
            main_logger.error(f"Layout analysis failed for {pdf_path}: {error_msg}")
            return f"Error during layout analysis: {error_msg}"
        if not layout_results:
             main_logger.error(f"Layout analysis returned no results for {pdf_path}.")
             return "Error: Layout analysis returned no results."


        structured_text = assistant.generate_structured_text_from_layout(layout_results)

        main_logger.info(f"Successfully processed PDF for structured text: {pdf_path} using wrapper.")
        return structured_text

    except Exception as e:
        logging.getLogger(__name__).error(f"Error in structured PDF extraction wrapper for PDF {pdf_path}: {e}", exc_info=True)
        return f"Error: An unexpected error occurred in the structured PDF processing wrapper: {str(e)}"
