import os
import unittest
from unittest.mock import patch, MagicMock
from dotenv import load_dotenv
from notion_manager import NotionManager

load_dotenv()  # Load environment variables from .env file

class TestNotionManager(unittest.TestCase):
    def setUp(self):
        self.notion_manager = NotionManager()

    @patch('notion_manager.NotionManager.notion')
    def test_create_page_in_database(self, mock_notion_client):
        # Test data
        md_text = "# Test Heading\nSample paragraph text."
        database_id = "mock_database_id"
        page_title = "Test Page"

        # Mock the response of pages.create
        mock_notion_client.pages.create.return_value = {'id': 'mock_page_id'}
        mock_notion_client.blocks.children.append.return_value = {}

        # Execute the method
        self.notion_manager.create_page_in_database(md_text, database_id, page_title)

        # Assertions
        mock_notion_client.pages.create.assert_called_once()
        mock_notion_client.blocks.children.append.assert_called()

    def test_md_to_notion_blocks_simple(self):
        md_text = "# Heading\n\nThis is a paragraph."
        blocks = self.notion_manager.md_to_notion_blocks(md_text)

        self.assertIsInstance(blocks, list)
        self.assertEqual(len(blocks), 2)
        self.assertEqual(blocks[0]['type'], 'heading_1')
        self.assertEqual(blocks[0]['heading_1']['rich_text'][0]['text']['content'], 'Heading')
        self.assertEqual(blocks[1]['type'], 'paragraph')
        self.assertEqual(blocks[1]['paragraph']['rich_text'][0]['text']['content'], 'This is a paragraph.')

    @patch.object(NotionManager, 'create_image_block')
    def test_md_to_notion_blocks_with_image(self, mock_create_image_block):
        md_text = "![Alt Text](./image.jpg)"
        mock_create_image_block.return_value = {'type': 'image'}
        
        blocks = self.notion_manager.md_to_notion_blocks(md_text)
        
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0]['type'], 'image')
        mock_create_image_block.assert_called_once_with('./image.jpg')

    @patch.object(NotionManager, 'create_page_in_database')
    def test_upload_chunks_to_database(self, mock_create_page):
        chunks = ["# Chunk 1 content", "# Chunk 2 content"]
        database_id = "mock_database_id"
        file_name = "test_file.pdf"

        self.notion_manager.upload_chunks_to_database(chunks, database_id, file_name)

        self.assertEqual(mock_create_page.call_count, 2)
        expected_calls = [
            (chunks[0], database_id, "test_file - Part 1"),
            (chunks[1], database_id, "test_file - Part 2")
        ]
        actual_calls = [call_args.args for call_args in mock_create_page.call_args_list]
        self.assertEqual(actual_calls, expected_calls)

    @patch.object(NotionManager, 'upload_image_and_get_url')
    @patch('os.path.exists', return_value=True)
    def test_create_image_block_valid(self, mock_exists, mock_upload_image):
        img_src = './image.jpg'
        mock_upload_image.return_value = 'http://example.com/image.jpg'

        image_block = self.notion_manager.create_image_block(img_src)

        self.assertIsNotNone(image_block)
        self.assertEqual(image_block['type'], 'image')
        self.assertEqual(image_block['image']['external']['url'], 'http://example.com/image.jpg')
        mock_upload_image.assert_called_once_with(os.path.abspath(img_src))

    @patch('os.path.exists', return_value=False)
    def test_create_image_block_invalid_path(self, mock_exists):
        img_src = './nonexistent.jpg'
        image_block = self.notion_manager.create_image_block(img_src)
        self.assertIsNone(image_block)

    @patch('requests.post')
    def test_upload_image_and_get_url(self, mock_post):
        img_path = 'test_image.jpg'

        # Create test image file
        with open(img_path, 'wb') as f:
            f.write(os.urandom(1024))

        try:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {'data': {'link': 'http://example.com/image.jpg'}}
            mock_post.return_value = mock_response

            image_url = self.notion_manager.upload_image_and_get_url(img_path)

            self.assertEqual(image_url, 'http://example.com/image.jpg')
            mock_post.assert_called_once()
        finally:
            # Clean up test image
            if os.path.exists(img_path):
                os.remove(img_path)

if __name__ == '__main__':
    unittest.main()
