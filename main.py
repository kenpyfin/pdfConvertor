import os
import logging
import argparse
from dotenv import load_dotenv
from pdf_extractor import extract_elements_combined, extract_images_to_debug
from notion_manager import NotionManager
import fitz

load_dotenv()  # Load environment variables from .env file

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Get and validate database ID from environment
notion_database_id = os.getenv('NOTION_DATABASE_ID')
if not notion_database_id:
    logger.error('NOTION_DATABASE_ID environment variable not set')
    exit(1)

def process_pdf_and_upload(file_path, database_id):
    """Process PDF and upload its content to Notion."""
    try:
        # # Extract images to debug folder
        # extract_images_to_debug(file_path)
        
        # Extract elements from PDF
        elements = extract_elements_combined(file_path)
        
        # Upload to Notion
        notion = NotionManager()
        notion.create_page_in_database(elements, database_id, os.path.basename(file_path))
        
        logger.info("PDF processing and page creation completed successfully")
        return True
    except Exception as e:
        logger.error(f"Error in process_pdf_and_upload: {e}")
        return False

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process PDF and upload to Notion')
    parser.add_argument('--pdf', required=True, help='Path to the PDF file')
    args = parser.parse_args()

    pdf_file = args.pdf
    success = process_pdf_and_upload(pdf_file, notion_database_id)
    if not success:
        exit(1)
