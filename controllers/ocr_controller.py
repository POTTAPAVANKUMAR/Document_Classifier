from flask import Blueprint, request, jsonify
from services.ocr_service import OCRService
import logging

ocr_bp = Blueprint('ocr', __name__)

@ocr_bp.route('/ocr', methods=['POST'])
def ocr():
    """
    Perform OCR on an image and return extracted text as a TXT file
    ---
    tags:
      - OCR
    parameters:
      - name: file
        in: formData
        type: file
        required: true
        description: The image file to perform OCR on
    responses:
      200:
        description: Text extracted from the image
        schema:
          properties:
            text:
              type: string
      400:
        description: Bad request if the file is not provided or is not an image
      500:
        description: Internal server error if an error occurs during OCR processing
    """
    try:
        if 'file' not in request.files:
            return jsonify({'message': 'No file found'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'message': 'No selected file'}), 400
        if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
            return jsonify({'message': 'Invalid file format'}), 400

        ocr_service = OCRService()
        text = ocr_service.perform_ocr(file)

        return jsonify({'text': text}), 200
    
    except Exception as e:
        logging.exception("Exception occurred during OCR processing")
        return jsonify({'message': str(e)}), 500
