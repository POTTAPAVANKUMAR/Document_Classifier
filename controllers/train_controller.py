from flask import Blueprint, request, jsonify
from services.training_service import TrainingService

train_bp = Blueprint('train', __name__)

@train_bp.route('/train', methods=['POST'])
def train():
    data = request.get_json()
    training_service = TrainingService()
    message = training_service.train_model(data)
    return jsonify({"message": message})
