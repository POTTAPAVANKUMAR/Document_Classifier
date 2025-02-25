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
    data = request.get_json().get('training_data', [])
    if not data:
        return jsonify({"message": "Invalid input data"}), 400

    training_service = TrainingService()
    try:
        message = training_service.train_model(data)
        return jsonify({"message": message})
    except ValueError as e:
        return jsonify({"message": str(e), "data": data}), 400
    except KeyError as e:
        return jsonify({"message": str(e)}), 400
    except Exception as e:
        return jsonify({"message": "Internal server error"}), 500