from flask import Flask
from controllers.train_controller import train_bp
from controllers.predict_controller import predict_bp

app = Flask(__name__)
app.register_blueprint(train_bp)
app.register_blueprint(predict_bp)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
