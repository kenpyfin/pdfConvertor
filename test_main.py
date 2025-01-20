import unittest
from unittest.mock import patch, MagicMock
from main import process_pdf_and_upload, reformat_markdown_with_claude
from dotenv import load_dotenv
import requests

load_dotenv()

class TestMain(unittest.TestCase):
    def test_reformat_markdown_with_claude_success(self):
        test_markdown = "# Test\nSome content"
        expected_output = "# Test\n\nSome content"
        
        mock_response = MagicMock()
        mock_response.json.return_value = {"formatted_text": expected_output}
        mock_response.raise_for_status.return_value = None
        
        with patch('requests.post') as mock_post:
            mock_post.return_value = mock_response
            with patch.dict('os.environ', {'CLAUDE_API_KEY': 'test_key'}):
                result = reformat_markdown_with_claude(test_markdown)
                
        self.assertEqual(result, expected_output)
        
    def test_reformat_markdown_with_claude_no_api_key(self):
        test_markdown = "# Test\nSome content"
        
        with patch.dict('os.environ', clear=True):
            result = reformat_markdown_with_claude(test_markdown)
            
        self.assertEqual(result, test_markdown)
        
    def test_reformat_markdown_with_claude_api_error(self):
        test_markdown = "# Test\nSome content"
        
        with patch('requests.post') as mock_post:
            mock_post.side_effect = requests.exceptions.RequestException()
            with patch.dict('os.environ', {'CLAUDE_API_KEY': 'test_key'}):
                result = reformat_markdown_with_claude(test_markdown)
                
        self.assertEqual(result, test_markdown)
    @patch('main.OCRPipe')  # Keep existing tests below this line
    @patch('main.NotionManager')
    def test_process_pdf_and_upload(self, mock_notion_manager_class, mock_ocr_pipe_class):
        # Mock OCRPipe instance
        mock_pipe_instance = MagicMock()
        mock_pipe_instance.pipe_classify.return_value = None
        mock_pipe_instance.pipe_analyze.return_value = None
        mock_pipe_instance.pipe_parse.return_value = None
        mock_pipe_instance.pipe_mk_markdown.return_value = "# Mocked Markdown Content"
        mock_ocr_pipe_class.return_value = mock_pipe_instance

        # Mock NotionManager instance
        mock_notion_manager_instance = MagicMock()
        mock_notion_manager_instance.upload_chunks_to_database.return_value = None
        mock_notion_manager_class.return_value = mock_notion_manager_instance

        # Execute the method
        result = process_pdf_and_upload('test.pdf', 'mock_database_id')

        # Assertions
        self.assertTrue(result)
        mock_pipe_instance.pipe_classify.assert_called_once()
        mock_pipe_instance.pipe_analyze.assert_called_once()
        mock_pipe_instance.pipe_parse.assert_called_once()
        mock_pipe_instance.pipe_mk_markdown.assert_called_once()
        mock_notion_manager_instance.upload_chunks_to_database.assert_called_once()

if __name__ == '__main__':
    unittest.main()
