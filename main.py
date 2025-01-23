import os
import logging
import argparse
import sys
import requests
import tiktoken

if sys.version_info < (3, 10):
    sys.exit("Python 3.10 or higher is required.")

import torch
from dotenv import load_dotenv
from magic_pdf.data.data_reader_writer import FileBasedDataWriter, FileBasedDataReader
from magic_pdf.config.make_content_config import DropMode, MakeMode
from magic_pdf.pipe.OCRPipe import OCRPipe
from torch.cuda.amp import autocast
from notion_manager import NotionManager

# Define output directory
OUTPUT_DIR = 'output'

# Create output directory if it doesn't exist
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

load_dotenv()  # Load environment variables from .env file

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Get and validate database ID from environment
notion_database_id = os.getenv('NOTION_DATABASE_ID')
if not notion_database_id:
    logger.error('NOTION_DATABASE_ID environment variable not set')
    exit(1)


def reformat_markdown_with_claude(md_text):
    from anthropic import Anthropic

    client = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
    chunks = split_markdown_into_chunks(md_text)
    reformatted_chunks = []
    system_prompt = "You are a helpful assistant that reformats markdown text to be more readable and consistent. Only output the reformatted markdown without any other words."

    for chunk in chunks:
        message = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=4096,
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": f"Please reformat this markdown text to be more readable without adding or deleting a single word:\n\n{chunk}"
                }
            ]
        )
        reformatted_chunk = message.content[].text if message.content else ""
        reformatted_chunks.append(reformatted_chunk)
    
    # Combine all reformatted chunks
    return "\n\n".join(reformatted_chunks)

def split_markdown_for_reformatting(md_text: str, max_tokens: int = 3500) -> list:
    """Split markdown text into chunks based on token count for reformatting."""
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(md_text)
    total_tokens = len(tokens)

    chunks = []
    start = 0
    while start < total_tokens:
        end = min(start + max_tokens, total_tokens)
        chunk_tokens = tokens[start:end]
        chunk_text = enc.decode(chunk_tokens)
        chunks.append(chunk_text)
        start = end

    return chunks

def  split_markdown_into_chunks(md_text: str, max_chunk_size: int = 10000, max_chunks: int = 10) -> list:
    """Split markdown text into chunks based on character count for Notion upload."""
    chunks = []
    current_chunk = ''
    for line in md_text.splitlines(keepends=True):
        if len(current_chunk) + len(line) > max_chunk_size:
            chunks.append(current_chunk)
            current_chunk = ''
            if len(chunks) >= max_chunks:
                break
        current_chunk += line
    if current_chunk and len(chunks) < max_chunks:
        chunks.append(current_chunk)
    return chunks

def process_pdf_and_upload(file_path, database_id, title=None):
    """Process PDF using OCRPipe and upload its content to Notion."""
    try:
        # Read PDF bytes
        reader = FileBasedDataReader("")
        pdf_bytes = reader.read(file_path)

        # Configure CUDA if available
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
            torch.set_default_dtype(torch.float32)
            torch.set_default_tensor_type(torch.cuda.FloatTensor)

        # Initialize and run OCR pipeline
        with autocast(dtype=torch.float16):
            model_list = []
            image_writer = FileBasedDataWriter("output/images")
            pipe = OCRPipe(pdf_bytes, model_list, image_writer)
            pipe.pipe_classify()
            pipe.pipe_analyze()
            pipe.pipe_parse()

            # Generate markdown content
            md_content = pipe.pipe_mk_markdown(
                "./images",
                drop_mode=DropMode.NONE,
                md_make_mode=MakeMode.MM_MD
            )

        # Get the file name and determine title
        file_name = os.path.basename(file_path)
        if title is None:
            title = os.path.splitext(file_name)[0]
            
        # Save markdown content to a file
        markdown_file_path = os.path.join(OUTPUT_DIR, f"{title}.md")
        if isinstance(md_content, list):
            md_text = "\n".join(md_content)
        else:
            md_text = md_content

        # Split md_text into chunks for reformatting
        reformat_chunks = split_markdown_for_reformatting(md_text)

        # Reformat each chunk using Claude
        reformatted_chunks = []
        for chunk in reformat_chunks:
            reformatted_chunk = reformat_markdown_with_claude(chunk)
            reformatted_chunks.append(reformatted_chunk)

        # Combine reformatted chunks into full markdown text
        md_text = "\n\n".join(reformatted_chunks)

        # Save the reformatted markdown content
        with open(markdown_file_path, 'w', encoding='utf-8') as f:
            f.write(md_text)

        # Now split the reformatted md_text into chunks for Notion upload
        chunks = split_markdown_into_chunks(md_text, max_chunk_size=10000, max_chunks=10)

        # Upload each chunk to Notion as a separate page
        notion = NotionManager(images_base_path='output')
        notion.upload_chunks_to_database(chunks, database_id, title)

        logger.info("PDF processed and content uploaded to Notion successfully")
        return True
    except Exception as e:
        logger.error(f"Error in process_pdf_and_upload: {e}")
        return False

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process PDF and upload to Notion')
    parser.add_argument('--pdf', required=True, help='Path to the PDF file')
    parser.add_argument('--title', required=False, help='Optional title for the markdown file and Notion pages')
    args = parser.parse_args()

    pdf_file = args.pdf
    title = args.title
    success = process_pdf_and_upload(pdf_file, notion_database_id, title)
    if not success:
        exit(1)

