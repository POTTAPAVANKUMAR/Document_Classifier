Document Classifier Project structure:

financial_document_classifier/
│
├── app.py
├── controllers/
│   └── __init__.py
│   └── predict_controller.py
│   └── train_controller.py
├── services/
│   └── __init__.py
│   └── prediction_service.py
│   └── training_service.py
├── models/
│   └── __init__.py
│   └── custom_dataset.py
├── config/
│   └── __init__.py
│   └── config.py
├── logs/
│   └── app.log
└── requirements.txt


python -m venv venv

source venv/bin/activate - mac
venv\Scripts\activate -windows

pip install -r requirements.txt

python app.py


To train the model :

curl -X POST http://localhost:5000/train -H "Content-Type: application/json" -d @train_data.json
[
    {"text": "Document 1 text", "label": 0},
    {"text": "Document 2 text", "label": 1},
    {"text": "Document 3 text", "label": 2},
    {"text": "Document 4 text", "label": 3}
]


Prediction :

curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d '{"texts": ["New document text to classify"]}'



-------------------
Full documentation:
-------------------

Project Documentation: Financial Document Classifier API
Table of Contents
Introduction
Project Structure
Installation
Configuration
Running the Application
API Endpoints
Logging
Training and Prediction Workflow
Extending the Project
Conclusion
Introduction
The Financial Document Classifier API is a Flask-based web service designed to classify financial documents using a BERT model. The API includes endpoints for training the model and making predictions on new documents. The project adheres to the SOLID principles, ensuring a well-structured, maintainable, and extendable codebase.

Project Structure
The project is organized as follows:

markdown
Copy code
financial_document_classifier/
│
├── app.py
├── controllers/
│   └── __init__.py
│   └── predict_controller.py
│   └── train_controller.py
├── services/
│   └── __init__.py
│   └── prediction_service.py
│   └── training_service.py
├── models/
│   └── __init__.py
│   └── custom_dataset.py
├── config/
│   └── __init__.py
│   └── config.py
├── logs/
│   └── app.log
├── requirements.txt
└── logging.yaml
Installation
Clone the Repository

bash
Copy code
git clone <repository_url>
cd financial_document_classifier
Create a Virtual Environment

bash
Copy code
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
Install Dependencies

bash
Copy code
pip install -r requirements.txt
Configuration
The logging configuration is specified in the logging.yaml file:

yaml
Copy code
version: 1
disable_existing_loggers: False
formatters:
    simple:
        format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
handlers:
    console:
        class: logging.StreamHandler
        level: DEBUG
        formatter: simple
        stream: ext://sys.stdout
    file:
        class: logging.FileHandler
        level: DEBUG
        formatter: simple
        filename: logs/app.log
loggers:
    app:
        level: DEBUG
        handlers: [console, file]
        propagate: no
root:
    level: DEBUG
    handlers: [console, file]
Running the Application
Start the Flask Application

bash
Copy code
python app.py
API Endpoints
1. Train the Model
Endpoint: /train

Method: POST

Request Body: JSON array of documents with text and label fields.

Response: JSON message indicating success.

Example Request:

json
Copy code
[
    {"text": "Document 1 text", "label": 0},
    {"text": "Document 2 text", "label": 1},
    {"text": "Document 3 text", "label": 2},
    {"text": "Document 4 text", "label": 3}
]
Example Response:

json
Copy code
{
    "message": "Model trained successfully!"
}
2. Predict Document Classes
Endpoint: /predict

Method: POST

Request Body: JSON object with a texts field containing an array of document texts.

Response: JSON array of prediction scores for each class.

Example Request:

json
Copy code
{
    "texts": ["New document text to classify"]
}
Example Response:

json
Copy code
[
    [
        {"label": "LABEL_0", "score": 0.1},
        {"label": "LABEL_1", "score": 0.2},
        {"label": "LABEL_2", "score": 0.3},
        {"label": "LABEL_3", "score": 0.4}
    ]
]
Logging
Logging is configured using logging.yaml. Logs are written to both the console and logs/app.log.

Training and Prediction Workflow
Training

The client sends a POST request to the /train endpoint with training data.
The train_controller handles the request and calls TrainingService.
TrainingService processes the data, trains the BERT model, and saves the model and tokenizer.
Prediction

The client sends a POST request to the /predict endpoint with texts to classify.
The predict_controller handles the request and calls PredictionService.
PredictionService loads the trained model and tokenizer, then makes predictions on the input texts.
Extending the Project
To extend the project, you can add more services, controllers, or models as needed. Follow the existing structure to ensure the project remains well-organized and maintainable.

Conclusion
The Financial Document Classifier API provides a robust and scalable solution for classifying financial documents using a BERT model. By adhering to SOLID principles and implementing good logging practices, the project is designed to be maintainable and extendable for future enhancements.