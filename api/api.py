from flask import Flask, request, jsonify
from flask_cors import CORS
from api.RequestService import RequestService

app = Flask(__name__)
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})


@app.route('/api', methods=['POST'])
def index():
    text = request.json['text']
    request_service = RequestService(text)
    data_frame = request_service.convert_target_column()
    text = data_frame.iloc[0]['text']
    req = {'text': str(text), 'sentiment': 'positive'}
    return jsonify(req)
