from flask import Flask, request, jsonify
from flask_cors import CORS
from api.GetRandomTweet import GetRandomTweet
from api.RequestService import RequestService

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})


@app.route('/api', methods=['POST'])
def sentiment_analysis():
    """Get row text as request and returns the given text cleaned along with the decisions about sentiment
    classification from several ml models"""
    classification_dto = RequestService(request.json['text']).classify_text()
    return jsonify(classification_dto.get_response())


@app.route('/getRandomTweet', methods=['GET'])
def retrieve_random_tweet():
    """Returns one random tweet along with the sentiment label from the testing dataframe"""
    get_random_tweet = GetRandomTweet()
    tweet_dto = get_random_tweet.get_tweet()
    return jsonify(tweet_dto.get_response())
