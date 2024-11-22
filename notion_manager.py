import os
import math
import requests
from datetime import datetime
from notion_client import Client
import logging
from pdf_extractor import clean_and_format_text

logger = logging.getLogger(__name__)

class NotionManager:
    def __init__(self):
        self.notion = Client(auth=os.getenv('NOTION_API_KEY'))
        if not os.getenv('NOTION_API_KEY'):
            raise ValueError("NOTION_API_KEY environment variable not set")

    def upload_image_and_get_url(self, image_path: str) -> str:
        """
        Upload the image to an image hosting service and return the URL.

        Args:
            image_path (str): Path to the image file.

        Returns:
            str: Publicly accessible URL of the uploaded image.
        """
        client_id = os.getenv('IMGUR_CLIENT_ID')
        if not client_id:
            raise ValueError("IMGUR_CLIENT_ID environment variable not set")

        headers = {"Authorization": f"Client-ID {client_id}"}
        with open(image_path, 'rb') as img_file:
            data = {'image': img_file.read()}
        response = requests.post("https://api.imgur.com/3/upload", headers=headers, files=data)

        if response.status_code == 200:
            image_url = response.json()['data']['link']
            return image_url
        else:
            logger.error(f"Image upload failed: {response.json()}")
            raise Exception("Image upload failed")

    def create_page_in_database(self, elements: list, database_id: str, file_name: str):
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
            
            content_blocks = []
            for element in elements:
                if element['type'] == 'text':
                    # Clean and format text
                    text = clean_and_format_text(element['content'])
                    # Split text into chunks to comply with Notion's limits
                    max_length = 2000
                    text_chunks = [text[i:i+max_length] for i in range(0, len(text), max_length)]
                    for chunk in text_chunks:
                        content_blocks.append({
                            "object": "block",
                            "type": "paragraph",
                            "paragraph": {
                                "rich_text": [
                                    {
                                        "type": "text",
                                        "text": {
                                            "content": chunk
                                        }
                                    }
                                ]
                            }
                        })
                elif element['type'] == 'image':
                    image_path = element['content']
                    # Upload the image and get a public URL
                    image_url = self.upload_image_and_get_url(image_path)
                    # Create an image block
                    content_blocks.append({
                        "object": "block",
                        "type": "image",
                        "image": {
                            "type": "external",
                            "external": {
                                "url": image_url
                            }
                        }
                    })
                    # Clean up temporary image file
                    os.remove(image_path)
            
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
