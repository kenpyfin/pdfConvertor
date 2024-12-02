import os
import math
from datetime import datetime
from notion_client import Client
import logging

logger = logging.getLogger(__name__)

class NotionManager:
    def __init__(self):
        self.notion = Client(auth=os.getenv('NOTION_API_KEY'))
        if not os.getenv('NOTION_API_KEY'):
            raise ValueError("NOTION_API_KEY environment variable not set")

    def upload_chunks_to_database(self, chunks, database_id, file_name):
        """Upload multiple chunks as separate pages to the Notion database."""
        for index, chunk_text in enumerate(chunks):
            page_title = f"{os.path.splitext(file_name)[0]} - Part {index + 1}"
            self.create_page_in_database(chunk_text, database_id, page_title)

    def create_page_in_database(self, text, database_id, page_title):
        """Create a new page in the Notion database with the extracted text."""
        try:
            logger.info(f"Creating a new page in Notion database {database_id}")
            # Define the properties of the new page
            properties = {
                "Name": {
                    "title": [
                        {
                            "type": "text",
                            "text": {
                                "content": page_title
                            }
                        }
                    ]
                },
            }
            
            # Create a single paragraph block with the text
            content_blocks = [{
                "object": "block",
                "type": "paragraph",
                "paragraph": {
                    "rich_text": [{
                        "type": "text",
                        "text": {"content": text}
                    }]
                }
            }]
            
            # Split content_blocks into batches of up to 100 blocks
            batch_size = 100
            total_batches = math.ceil(len(content_blocks) / batch_size)
            batches = [content_blocks[i * batch_size:(i + 1) * batch_size] for i in range(total_batches)]
            
            # Create the new page with the first batch of blocks
            response = self.notion.pages.create(
                parent={"database_id": database_id},
                properties=properties,
                children=batches[0]
            )
            page_id = response['id']
            
            # Append remaining batches if any
            for batch in batches[1:]:
                self.notion.blocks.children.append(
                    block_id=page_id,
                    children=batch
                )
            
            logger.info("Successfully created a new page in Notion database")
        except Exception as e:
            logger.error(f"Error creating a page in Notion database: {e}")
            raise
