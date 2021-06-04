from flask import Flask, request, jsonify

from api.RequestService import RequestService

app = Flask(__name__)


@app.route('/', methods=['POST'])
def index():
    text = request.json['text']
    request_service = RequestService(text)
    data_frame = request_service.convert_target_column()
    text = data_frame.iloc[0]['text']
    req = {'text': str(text)}
    return jsonify(req)
