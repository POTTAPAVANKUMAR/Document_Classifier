from PIL import Image
import pytesseract
import logging
from .base_service import BaseService

logger = logging.getLogger(__name__)

class OCRService(BaseService):
    def perform_ocr(self, file):
        try:
            logger.info("Performing OCR on the provided image file")
            img = Image.open(file)
            text = pytesseract.image_to_string(img)
            return text
        except Exception as e:
            logger.error("Error occurred during OCR processing", exc_info=True)
            raise e
