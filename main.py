import os
import logging
import argparse
from dotenv import load_dotenv
from pdf_extractor import extract_text_combined
from notion_manager import NotionManager

# Define output directory
OUTPUT_DIR = 'output'

# Create output directory if it doesn't exist
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

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
        # Extract text from PDF
        text = extract_text_combined(file_path)
        
        # Upload the entire text to Notion as a single page
        notion = NotionManager()
        notion.create_page_in_database(text, database_id, os.path.basename(file_path))
        
        logger.info("PDF processed and content uploaded to Notion successfully")
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
