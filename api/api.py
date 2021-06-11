from flask import Flask, request, jsonify
from flask_cors import CORS
from api.RequestService import RequestService

app = Flask(__name__)
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})


@app.route('/api', methods=['POST'])
def index():
    text = request.json['text']
    request_service = RequestService(text)
    ds = request_service.convert_target_column()
    data_frame = ds.get_dataframe()
    text = data_frame.iloc[0]['text']
    print(ds.get_array())
    arr = ds.get_array()[0]
    print(arr)
    req = {'text': str(text),
           'neutral': str(arr[0]),
           'negative': str(arr[1]),
           'positive': str(arr[2])

           }
    return jsonify(req)
