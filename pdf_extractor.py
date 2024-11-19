import os
from pdfminer.high_level import extract_text
from pdf2image import convert_from_path
import pytesseract

# Configure Tesseract path if needed
if os.name == 'nt':  # Windows
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
import logging

logger = logging.getLogger(__name__)

def extract_text_from_pdf(file_path):
    """Extract text from PDFs with embedded text."""
    try:
        text = extract_text(file_path)
        return text
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e}")
        raise

def extract_text_with_ocr(file_path):
    """Extract text from scanned PDFs using OCR."""
    try:
        images = convert_from_path(file_path)
        text = ''
        for image in images:
            text += pytesseract.image_to_string(image)
        return text
    except Exception as e:
        logger.error(f"Error performing OCR: {e}")
        raise

def extract_text_combined(file_path):
    """Determine and use appropriate extraction method."""
    logger.info(f"Starting text extraction from {file_path}")
    try:
        text = extract_text_from_pdf(file_path)
        if not text.strip():
            logger.info("No embedded text found, attempting OCR")
            text = extract_text_with_ocr(file_path)
            if not text.strip():
                logger.error("No text extracted from OCR")
                raise ValueError("Text extraction failed")
        return text
    except Exception as e:
        logger.error(f"Error in combined text extraction: {e}")
        raise
