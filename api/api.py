from flask import Flask, request, jsonify
from flask_cors import CORS
from api.RequestService import RequestService

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})


@app.route('/api', methods=['POST'])
def index():
    text = request.json['text']
    request_service = RequestService(text)
    classification_dto = request_service.classify_text()
    return jsonify(classification_dto.get_response())
