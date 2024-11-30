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

    def create_page_in_database(self, text, database_id, file_name):
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
                                "content": file_name
                            }
                        }
                    ]
                },
            }
            
            # Parse text into lines and create appropriate blocks
            lines = text.split('\n')
            content_blocks = []
            
            for line in lines:
                line = line.strip()
                if line.startswith('# '):
                    # Heading 1
                    content_blocks.append({
                        "object": "block",
                        "type": "heading_1",
                        "heading_1": {
                            "rich_text": [{
                                "type": "text",
                                "text": {"content": line[2:].strip()}
                            }]
                        }
                    })
                elif line.startswith('## '):
                    # Heading 2
                    content_blocks.append({
                        "object": "block",
                        "type": "heading_2",
                        "heading_2": {
                            "rich_text": [{
                                "type": "text",
                                "text": {"content": line[3:].strip()}
                            }]
                        }
                    })
                elif line.startswith('- '):
                    # Bulleted list item
                    content_blocks.append({
                        "object": "block",
                        "type": "bulleted_list_item",
                        "bulleted_list_item": {
                            "rich_text": [{
                                "type": "text",
                                "text": {"content": line[2:].strip()}
                            }]
                        }
                    })
                elif line:
                    # Regular paragraph
                    content_blocks.append({
                        "object": "block",
                        "type": "paragraph",
                        "paragraph": {
                            "rich_text": [{
                                "type": "text",
                                "text": {"content": line}
                            }]
                        }
                    })
            
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
