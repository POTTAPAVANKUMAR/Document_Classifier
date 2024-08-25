from flask import Blueprint, request, jsonify
from services.training_service import TrainingService

train_bp = Blueprint('train', __name__)

@train_bp.route('/train', methods=['POST'])
def train():
    """
    Train the model with the provided data
    ---
    tags:
      - Training
    parameters:
      - name: body
        in: body
        required: true
        description: JSON object containing training data
        schema:
          type: object
          properties:
            training_data:
              type: array
              items:
                type: object
                properties:
                  text:
                    type: string
                    example: "Sample training text"
                  label:
                    type: string
                    example: "Label for the text"
    responses:
      200:
        description: Training completed successfully
        schema:
          type: object
          properties:
            message:
              type: string
              example: "Training completed successfully."
      400:
        description: Invalid input data
      500:
        description: Internal server error
    """
    data = request.get_json()
    training_service = TrainingService()
    message = training_service.train_model(data)
    return jsonify({"message": message})
