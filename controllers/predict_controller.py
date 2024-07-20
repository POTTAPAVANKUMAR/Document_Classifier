from flask import Blueprint, request, jsonify
from services.prediction_service import PredictionService

predict_bp = Blueprint('predict', __name__)

@predict_bp.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    texts = data.get('texts', [])
    prediction_service = PredictionService()
    predictions = prediction_service.predict(texts)
    return jsonify(predictions)
