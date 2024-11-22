import os
import time
import fitz  # PyMuPDF
from pdfminer.high_level import extract_text, extract_pages
from pdfminer.layout import LAParams, LTTextBox, LTTextLine, LTImage
from pdf2image import convert_from_path
import pytesseract
import logging
import tempfile
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import tensorflow as tf
import tensorflow_hub as hub

# Debug image folder configuration
DEBUG_IMAGE_FOLDER = 'debug_images'

# Configure Tesseract path if needed
if os.name == 'nt':  # Windows
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

logger = logging.getLogger(__name__)

def clean_and_format_text(text: str) -> str:
    """
    Perform basic trimming and cleanup on the extracted text.

    This function removes leading and trailing whitespace from each line,
    replaces multiple spaces with a single space,
    and removes extra blank lines.

    Args:
        text (str): The extracted text.

    Returns:
        str: The cleaned and formatted text.
    """
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        # Strip leading and trailing whitespace
        line = line.strip()

        # Replace multiple spaces with a single space
        line = ' '.join(line.split())

        # Skip empty lines
        if not line:
            continue

        cleaned_lines.append(line)

    cleaned_text = '\n'.join(cleaned_lines)
    return cleaned_text

def extract_elements_with_positions(file_path: str) -> list:
    """
    Extract text and images from the PDF, preserving their order.

    Args:
        file_path (str): Path to the PDF file.

    Returns:
        list: A list of elements where each element is a dictionary containing
              the type ('text' or 'image') and content.
    """
    if not os.path.exists(DEBUG_IMAGE_FOLDER):
        os.makedirs(DEBUG_IMAGE_FOLDER)
        
    elements = []
    try:
        laparams = LAParams()
        for page_layout in extract_pages(file_path, laparams=laparams):
            for element in page_layout:
                if isinstance(element, (LTTextBox, LTTextLine)):
                    text = element.get_text()
                    if text.strip():
                        elements.append({
                            'type': 'text',
                            'content': text
                        })
                elif isinstance(element, LTImage):
                    # Save the image to a temporary file
                    img_stream = element.stream.get_rawdata()
                    if img_stream:
                        timestamp = int(time.time())
                        debug_image_path = os.path.join(DEBUG_IMAGE_FOLDER, f'extracted_image_{timestamp}.png')
                        with open(debug_image_path, 'wb') as debug_file:
                            debug_file.write(img_stream)
                        
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as img_file:
                            img_file.write(img_stream)
                            img_path = img_file.name
                        elements.append({
                            'type': 'image',
                            'content': img_path
                        })
        return elements
    except Exception as e:
        logger.error(f"Error extracting elements from PDF: {e}")
        raise

def extract_elements_with_ocr(file_path: str) -> list:
    """
    Extract text and images from scanned PDFs using OCR.

    Args:
        file_path (str): Path to the PDF file.

    Returns:
        list: A list of elements where each element is a dictionary containing
              the type ('text' or 'image') and content.
    """
    if not os.path.exists(DEBUG_IMAGE_FOLDER):
        os.makedirs(DEBUG_IMAGE_FOLDER)
        
    try:
        images = convert_from_path(file_path)
        elements = []
        for image in images:
            # Save the image temporarily
            timestamp = int(time.time())
            debug_image_path = os.path.join(DEBUG_IMAGE_FOLDER, f'scanned_image_{timestamp}.png')
            image.save(debug_image_path, 'PNG')
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as img_file:
                image.save(img_file.name, 'PNG')
                img_path = img_file.name
            elements.append({
                'type': 'image',
                'content': img_path
            })
            # Extract text via OCR
            text = pytesseract.image_to_string(image)
            if text.strip():
                elements.append({
                    'type': 'text',
                    'content': text
                })
        return elements
    except Exception as e:
        logger.error(f"Error performing OCR: {e}")
        raise


def detect_chart_regions_yolo(img_array):
    """
    Detect charts using YOLOv8 pre-trained model.
    """
    try:
        # Load the pre-trained model
        model = YOLO('yolov8n.pt')  # or use a custom trained model for charts
        
        # Run detection
        results = model(img_array)
        
        # Process results
        detected_regions = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get confidence score
                confidence = float(box.conf)
                # Get class
                class_id = int(box.cls)
                # Get coordinates (convert to int)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Filter based on confidence and class
                if confidence > 0.5:  # Adjust confidence threshold as needed
                    detected_regions.append((x1, y1, x2, y2))
        
        return detected_regions if detected_regions else None
        
    except Exception as e:
        logger.error(f"Error in YOLO detection: {e}")
        return None



def detect_chart_regions_tensorflow(img_array):
    """
    Detect charts using TensorFlow pre-trained model.
    """
    try:
        # Load pre-trained model
        detector = hub.load("https://tfhub.dev/tensorflow/faster_rcnn/inception_resnet_v2_640x640/1")
        
        # Prepare image
        input_image = tf.convert_to_tensor(img_array)
        input_image = tf.expand_dims(input_image, 0)
        
        # Run detection
        results = detector(input_image)
        
        # Process results
        detected_regions = []
        boxes = results["detection_boxes"][0].numpy()
        scores = results["detection_scores"][0].numpy()
        
        height, width = img_array.shape[:2]
        
        for i, score in enumerate(scores):
            if score > 0.5:  # Confidence threshold
                # Convert normalized coordinates to pixel coordinates
                y1, x1, y2, x2 = boxes[i]
                x1 = int(x1 * width)
                x2 = int(x2 * width)
                y1 = int(y1 * height)
                y2 = int(y2 * height)
                
                detected_regions.append((x1, y1, x2, y2))
        
        return detected_regions if detected_regions else None
        
    except Exception as e:
        logger.error(f"Error in TensorFlow detection: {e}")
        return None

def detect_chart_regions(img_array):
    """
    Enhanced chart detection using multiple characteristics:
    - Axis lines detection
    - Grid pattern recognition
    - Data point/line detection
    - Text density analysis
    """
    
    # Convert to grayscale if not already
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    
    # Create binary images with different thresholds
    _, binary_strict = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)  # For lines
    _, binary_loose = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)   # For data points
    
    # Edge detection with different parameters
    edges_strict = cv2.Canny(binary_strict, 50, 150)
    edges_loose = cv2.Canny(binary_loose, 30, 100)
    
    # Find horizontal and vertical lines (potential axes and grid)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
    
    horizontal_lines = cv2.morphologyEx(binary_strict, cv2.MORPH_OPEN, horizontal_kernel)
    vertical_lines = cv2.morphologyEx(binary_strict, cv2.MORPH_OPEN, vertical_kernel)
    
    # Combine detected lines
    combined_lines = cv2.addWeighted(horizontal_lines, 0.5, vertical_lines, 0.5, 0)
    
    # Detect strong lines using Hough transform
    lines = cv2.HoughLinesP(
        combined_lines, 
        rho=1,
        theta=np.pi/180,
        threshold=50,
        minLineLength=50,
        maxLineGap=10
    )
    
    if lines is None:
        return None
    
    # Find contours for potential data points and chart elements
    contours, _ = cv2.findContours(binary_loose, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Analyze line intersections and contour positions
    x_coords = []
    y_coords = []
    
    # Add line coordinates
    for line in lines:
        x1, y1, x2, y2 = line[0]
        x_coords.extend([x1, x2])
        y_coords.extend([y1, y2])
    
    # Add contour coordinates for significant contours
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 10:  # Filter out noise
            x, y, w, h = cv2.boundingRect(contour)
            x_coords.extend([x, x + w])
            y_coords.extend([y, y + h])
    
    if not x_coords or not y_coords:
        return None
    
    # Calculate boundaries
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    
    # Validate chart region
    width = x_max - x_min
    height = y_max - y_min
    
    # Size validation
    min_size = 100
    if width < min_size or height < min_size:
        return None
    
    # Aspect ratio validation (charts usually aren't extremely narrow or wide)
    aspect_ratio = width / height
    if aspect_ratio < 0.2 or aspect_ratio > 5:
        return None
    
    # Add padding
    padding = 20
    x_min = max(0, x_min - padding)
    x_max = min(img_array.shape[1], x_max + padding)
    y_min = max(0, y_min - padding)
    y_max = min(img_array.shape[0], y_max + padding)
    
    # Debug visualization (uncomment if needed)
    """
    debug_img = img_array.copy()
    cv2.rectangle(debug_img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    cv2.imwrite('debug_detection.png', debug_img)
    """
    
    return (x_min, y_min, x_max, y_max)

def extract_images_to_debug(file_path):
    """Extract chart images from PDF to debug folder using intelligent detection."""
    pdf_file = fitz.open(file_path)
    debug_dir = "debug_images"
    os.makedirs(debug_dir, exist_ok=True)
    
    success = False
    
    for page_index in range(len(pdf_file)):
        try:
            page = pdf_file[page_index]
            
            # Get high-resolution page image
            zoom = 2
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)
            
            # Convert to PIL Image and then numpy array
            img_data = pix.samples
            img = Image.frombytes("RGB", [pix.width, pix.height], img_data)
            img_array = np.array(img)
            
            # Detect chart regions
            chart_bounds = detect_chart_regions_tensorflow(img_array)
            
            if chart_bounds is not None:
                x_min, y_min, x_max, y_max = chart_bounds
                
                # Crop the chart region
                chart_image = img.crop((x_min, y_min, x_max, y_max))
                
                # Save the chart
                image_name = f"chart_page{page_index + 1}.png"
                image_path = os.path.join(debug_dir, image_name)
                chart_image.save(image_path, 'PNG')
                
                logger.debug(f"Saved chart: {image_name}")
                success = True
            
        except Exception as e:
            logger.error(f"Error extracting chart from page {page_index + 1}: {e}")
            continue
    
    pdf_file.close()
    return success

def extract_elements_combined(file_path: str) -> list:
    """Extract elements (text and images) from the PDF, handling both text PDFs and scanned PDFs."""
    logger.info(f"Starting element extraction from {file_path}")
    try:
        elements = extract_elements_with_positions(file_path)
        if not elements:
            logger.info("No elements found with PDFMiner, attempting OCR")
            elements = extract_elements_with_ocr(file_path)
            if not elements:
                logger.error("No elements extracted from OCR")
                raise ValueError("Element extraction failed")
        return elements
    except Exception as e:
        logger.error(f"Error in combined element extraction: {e}")
        raise
