from flask import Blueprint, request, jsonify
from services.prediction_service import PredictionService

predict_bp = Blueprint('predict', __name__)

@predict_bp.route('/predict', methods=['POST'])
def predict():
    """
    Predict the labels for the input texts
    ---
    tags:
      - Prediction
    parameters:
      - name: body
        in: body
        required: true
        description: JSON object containing texts for prediction
        schema:
          type: object
          required:
            - texts
          properties:
            texts:
              type: array
              items:
                type: string
              example: ["Text 1", "Text 2"]
    responses:
      200:
        description: Predictions for the input texts
        schema:
          type: array
          items:
            type: object
            properties:
              label:
                type: string
                description: The predicted label
              score:
                type: number
                format: float
                description: Confidence score for the prediction
      400:
        description: Invalid input data
      500:
        description: Internal server error
    """
    data = request.get_json()
    texts = data.get('texts', [])
    prediction_service = PredictionService()
    predictions = prediction_service.predict(texts)
    return jsonify(predictions)
