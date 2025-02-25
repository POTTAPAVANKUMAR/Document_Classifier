import logging
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib
from .base_service import BaseService

logger = logging.getLogger(__name__)

class TrainingService(BaseService):
    def __init__(self):
        super().__init__()
        self.vectorizer = TfidfVectorizer()
        self.model = LogisticRegression()

    def train_model(self, data):
        df = pd.DataFrame(data)

        # Ensure enough data is available
        if len(df) < 2:
            logger.error("Not enough data to perform train-test split. At least 2 samples are required.")
            raise ValueError("Not enough data to perform train-test split. At least 2 samples are required.")

        # Check if required columns are present
        if 'text' not in df.columns or 'label' not in df.columns:
            logger.error(f"DataFrame columns: {df.columns}")
            raise KeyError("DataFrame must contain 'text' and 'label' columns.")

        # Train-test split (90% train, 10% test)
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            df['text'], df['label'], test_size=0.1, random_state=42
        )

        # Convert text to TF-IDF features
        X_train = self.vectorizer.fit_transform(train_texts)
        X_val = self.vectorizer.transform(val_texts)

        # Train Logistic Regression model
        self.model.fit(X_train, train_labels)

        # Evaluate model accuracy (only if val_texts exist)
        if len(val_texts) > 0:
            accuracy = self.model.score(X_val, val_labels)
            logger.info(f"Validation Accuracy: {accuracy:.2f}")

        # Ensure the models directory exists
        os.makedirs("models", exist_ok=True)

        # Save the trained model and vectorizer
        joblib.dump(self.model, "models/logistic_regression.pkl")
        joblib.dump(self.vectorizer, "models/vectorizer.pkl")

        logger.info("Model trained and saved successfully.")
        return "Model trained successfully!"

    def predict(self, text):
        # Load model and vectorizer if not already initialized
        try:
            model = joblib.load("models/logistic_regression.pkl")
            vectorizer = joblib.load("models/vectorizer.pkl")
        except Exception as e:
            logger.error("Failed to load the trained model. Train the model first.")
            raise ValueError("No trained model found. Please train the model first.")

        # Convert text to numerical format
        text_vectorized = vectorizer.transform([text])
        
        # Predict
        prediction = model.predict(text_vectorized)[0]
        return prediction