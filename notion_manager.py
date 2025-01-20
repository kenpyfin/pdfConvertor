import os
import requests
from notion_client import Client
import logging
import markdown
import bs4

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class NotionManager:
    def __init__(self, images_base_path=''):
        self.notion = Client(auth=os.getenv('NOTION_API_KEY'))
        if not os.getenv('NOTION_API_KEY'):
            raise ValueError("NOTION_API_KEY environment variable not set")
        self.images_base_path = images_base_path

    def upload_chunks_to_database(self, chunks, database_id, title):
        """Upload chunks as subpages under a main page in the Notion database."""
        try:
            # Create the parent page in the database
            parent_page_response = self.notion.pages.create(
                parent={"database_id": database_id},
                properties={
                    "Name": {
                        "title": [
                            {
                                "type": "text",
                                "text": {
                                    "content": title
                                }
                            }
                        ]
                    },
                }
            )
            parent_page_id = parent_page_response['id']
            logger.info(f"Created parent page '{title}' with ID {parent_page_id}")

            # For each chunk, create a subpage under the parent page
            for index, chunk_text in enumerate(chunks):
                subpage_title = f"Part {index + 1}"
                self.create_subpage_in_page(chunk_text, parent_page_id, subpage_title)
            
            logger.info("Successfully uploaded all chunks as subpages under the main page.")
        except Exception as e:
            logger.error(f"Error uploading chunks to Notion: {e}")
            raise

    def md_to_notion_blocks(self, md_text):
        """
        Convert Markdown text to Notion block objects.
        """
        html = markdown.markdown(md_text, extensions=['markdown.extensions.extra'])
        logger.debug(f"Converted HTML: {html}")
        soup = bs4.BeautifulSoup(html, features="html.parser")

        def parse_element(element):
            blocks = []
            for child in element.contents:
                if isinstance(child, bs4.Tag):
                    logger.debug(f"Processing HTML tag: {child.name}")
                    if child.name in ['h1', 'h2', 'h3']:
                        heading_text = child.get_text()
                        while heading_text:
                            chunk_text = heading_text[:2000]
                            heading_text = heading_text[2000:]
                            blocks.append({
                                f"type": f"heading_{child.name[1]}",
                                f"heading_{child.name[1]}": {
                                    "rich_text": [{"type": "text", "text": {"content": chunk_text}}]
                                }
                            })
                    elif child.name == 'p':
                        paragraph_blocks = parse_paragraph(child)
                        blocks.extend(paragraph_blocks)
                    elif child.name in ['ul', 'ol']:
                        blocks.extend(parse_list(child, "bulleted_list_item" if child.name == 'ul' else "numbered_list_item"))
                    elif child.name == 'img':
                        img_src = child.get('src')
                        if img_src:
                            image_block = self.create_image_block(img_src)
                            if image_block:
                                blocks.append(image_block)
                    else:
                        blocks.extend(parse_element(child))
                elif isinstance(child, bs4.NavigableString):
                    text = str(child).strip()
                    if text:
                        blocks.append({
                            "type": "paragraph",
                            "paragraph": {"rich_text": [{"type": "text", "text": {"content": text}}]}
                        })
            return blocks

        def parse_paragraph(element):
            blocks = []
            for child in element.contents:
                if isinstance(child, bs4.Tag):
                    if child.name == 'img':
                        logger.debug(f"Found image tag in paragraph")
                        img_src = child.get('src')
                        if img_src:
                            image_block = self.create_image_block(img_src)
                            if image_block:
                                blocks.append(image_block)
                    else:
                        blocks.extend(parse_element(child))
                elif isinstance(child, bs4.NavigableString):
                    text = str(child).strip()
                    if text:
                        blocks.append({
                            "type": "paragraph",
                            "paragraph": {"rich_text": [{"type": "text", "text": {"content": text}}]}
                        })
            return blocks

        def parse_list(element, list_item_type):
            items = []
            for li in element.find_all('li', recursive=False):
                blocks = []
                for child in li.contents:
                    if isinstance(child, bs4.Tag):
                        blocks.extend(parse_element(child))
                    elif isinstance(child, bs4.NavigableString):
                        text = str(child).strip()
                        if text:
                            blocks.append({
                                "type": list_item_type,
                                list_item_type: {"rich_text": [{"type": "text", "text": {"content": text}}]}
                            })
                items.extend(blocks)
            return items

        return parse_element(soup)

    def create_subpage_in_page(self, md_text, parent_page_id, page_title):
        """Create a new subpage under the specified parent page with the markdown content."""
        try:
            logger.info(f"Creating subpage '{page_title}' under parent page ID {parent_page_id}")

            # Create a new subpage under the parent page
            response = self.notion.pages.create(
                parent={"page_id": parent_page_id},
                properties={
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
            )
            page_id = response['id']

            # Convert markdown to Notion blocks
            notion_blocks = self.md_to_notion_blocks(md_text)

            # Append blocks in chunks of 100 to comply with Notion API limits
            for i in range(0, len(notion_blocks), 100):
                chunk = notion_blocks[i:i + 100]
                self.notion.blocks.children.append(
                    block_id=page_id,
                    children=chunk
                )
            
            logger.info(f"Successfully created subpage '{page_title}' with ID {page_id}")
        except Exception as e:
            logger.error(f"Error creating subpage '{page_title}': {e}")
            raise

    def create_image_block(self, img_src):
        """
        Create an image block in Notion from the given image source.

        Args:
            img_src (str): The source path of the image.

        Returns:
            dict: A Notion image block object.
        """
        try:
            # Resolve the image path
            logger.debug(f"Original image source: {img_src}")
            img_src = os.path.normpath(os.path.join(self.images_base_path, img_src.lstrip("./")))
            logger.debug(f"Resolved image path: {img_src}")

            if not os.path.exists(os.path.abspath(img_src)):
                logger.warning(f"Image file not found: {img_src}")
                return None

            # Upload the image and get a publicly accessible URL
            logger.debug(f"Uploading image: {img_src}")
            image_url = self.upload_image_and_get_url(img_src)

            if not image_url:
                logger.error(f"Failed to upload image: {img_src}")
                return None

            # Create an image block with the URL
            image_block = {
                "type": "image",
                "image": {
                    "type": "external",
                    "external": {
                        "url": image_url
                    }
                }
            }
            return image_block

        except Exception as e:
            logger.error(f"Error creating image block for {img_src}: {e}")
            return None

    def upload_image_and_get_url(self, img_path):
        """
        Upload the image to the new image server and return the public URL.
        """
        try:
            upload_url = 'https://image.kximaginary.com/upload'
            api_key = os.getenv('IMAGE_API_KEY')
            
            if not api_key:
                logger.error("IMAGE_API_KEY not found in environment variables")
                return None
                
            with open(img_path, 'rb') as f:
                files = {'file': f}
                params = {'key': api_key}
                response = requests.post(upload_url, files=files, params=params)

            if response.status_code == 200:
                response_data = response.json()
                image_url = response_data.get('url')
                return image_url
            else:
                logger.error(f"Image upload failed with status {response.status_code}: {response.text}")
                return None
        except Exception as e:
            logger.error(f"Error uploading image {img_path}: {str(e)}")
            return None

    def _get_content_type(self, file_path):
        """
        Get the content type based on file extension.
        """
        extension = os.path.splitext(file_path)[1].lower()
        content_types = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.webp': 'image/webp'
        }
        return content_types.get(extension, 'application/octet-stream')
