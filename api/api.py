from flask import Flask, request, jsonify
from flask_cors import CORS
from api.RequestService import RequestService
from api.GetRandomTweet import GetRandomTweet

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})


@app.route('/api', methods=['POST'])
def sentiment_analysis():
    requested_text = request.json['text']  # get requested text
    request_service = RequestService(requested_text)
    classification_dto = request_service.classify_text()
    return jsonify(classification_dto.get_response())


@app.route('/getRandomTweet', methods=['GET'])
def get_random_tweet():
    get_random_tweet = GetRandomTweet()
    tweet_dto = get_random_tweet.get_tweet()
    return jsonify(tweet_dto.get_response())
