import unittest
from unittest.mock import patch, MagicMock
from main import process_pdf_and_upload
from dotenv import load_dotenv

load_dotenv()

class TestMain(unittest.TestCase):
    @patch('main.OCRPipe')
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
