import os
import logging
import argparse
from dotenv import load_dotenv
from transformers import pipeline
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


def process_text_with_llm(text):
    """Use an on-device LLM to format text and divide into meaningful chunks."""
    # Initialize the summarization pipeline with an on-device model
    summarizer = pipeline("summarization", model='sshleifer/distilbart-cnn-12-6', device=-1)  # Use CPU (-1)

    # Split the text into manageable chunks
    max_chunk_size = 500  # Max tokens per chunk (adjust as needed)
    text_length = len(text)
    chunks = [text[i:i + max_chunk_size] for i in range(0, text_length, max_chunk_size)]

    # Process each chunk with the LLM
    processed_chunks = []
    for chunk in chunks:
        summary = summarizer(chunk, max_length=130, min_length=30, do_sample=False)
        processed_chunks.append(summary[0]['summary_text'])

    return processed_chunks

def process_pdf_and_upload(file_path, database_id):
    """Process PDF and upload its content to Notion."""
    try:
        # Extract text from PDF
        text = extract_text_combined(file_path)
        
        # Save the entire text to a markdown file
        file_name = os.path.basename(file_path)
        markdown_file_path = os.path.join(OUTPUT_DIR, f"{os.path.splitext(file_name)[0]}.md")
        with open(markdown_file_path, 'w', encoding='utf-8') as f:
            f.write(text)

        # Process text with the LLM
        chunks = process_text_with_llm(text)

        # Upload each chunk to Notion as a separate page
        notion = NotionManager()
        notion.upload_chunks_to_database(chunks, database_id, file_name)
        
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
