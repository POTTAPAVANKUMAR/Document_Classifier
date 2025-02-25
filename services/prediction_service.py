import logging
import joblib
from .base_service import BaseService

logger = logging.getLogger(__name__)

class PredictionService(BaseService):
    def __init__(self):
        try:
            # Load trained model and vectorizer
            self.model = joblib.load("models/logistic_regression.pkl")
            self.vectorizer = joblib.load("models/vectorizer.pkl")
            logger.info("Model and vectorizer loaded successfully.")
        except Exception as e:
            logger.error("Failed to load the trained model. Train the model first.")
            raise ValueError("No trained model found. Please train the model first.")

    def predict(self, texts):
        if isinstance(texts, str):
            texts = [texts]  # Convert single string input to a list

        logger.info("Predicting classes for input texts")
        
        # Convert texts to numerical features
        text_vectorized = self.vectorizer.transform(texts)
        
        # Predict
        predictions = self.model.predict(text_vectorized)
        
        return predictions.tolist()
