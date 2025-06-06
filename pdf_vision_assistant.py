import os
import fitz  # PyMuPDF
from paddleocr import PaddleOCR, PPStructure
import logging

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
            # For layout analysis, table recognition, etc.
            # self.structure_analyzer = PPStructure(lang=lang, show_log=False, **kwargs) # TODO: Enable if structure analysis is needed
            logger.info("PaddleOCR initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize PaddleOCR: {e}", exc_info=True)
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
