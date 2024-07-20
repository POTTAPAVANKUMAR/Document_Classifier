import logging
from transformers import BertTokenizer, BertForSequenceClassification, TextClassificationPipeline

logger = logging.getLogger(__name__)

class PredictionService:
    def __init__(self):
        self.model = BertForSequenceClassification.from_pretrained('./model')
        self.tokenizer = BertTokenizer.from_pretrained('./model')
        self.classifier = TextClassificationPipeline(model=self.model, tokenizer=self.tokenizer, return_all_scores=True)

    def predict(self, texts):
        logger.info("Predicting classes for input texts")
        predictions = self.classifier(texts)
        return predictions
