import os
import logging
from dotenv import load_dotenv
from notion_manager import NotionManager

load_dotenv()  # Load environment variables from .env file

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_upload_image_and_get_url():
    # Instantiate NotionManager
    notion_manager = NotionManager()
    
    # Path to the test image
    test_image_path = 'output/images/0a073017d1c595417324c0cfbfb794dadfc48b892f65a5e4def261ce7df91937.jpg'  # Replace with the actual path to your test image
    
    # Check if the test image exists
    if not os.path.exists(test_image_path):
        logger.error(f"Test image not found at {test_image_path}")
        print("Test image not found.")
        return
    
    try:
        # Upload the image
        image_url = notion_manager.upload_image_and_get_url(test_image_path)
        
        # Check the result
        if image_url:
            logger.info(f"Image uploaded successfully. URL: {image_url}")
            print(f"Test passed. Image URL: {image_url}")
        else:
            logger.error("Image upload failed.")
            print("Test failed. Image upload returned None.")
    except Exception as e:
        logger.error(f"An error occurred during the test: {e}")
        print(f"Test failed with exception: {e}")

if __name__ == '__main__':
    test_upload_image_and_get_url()
