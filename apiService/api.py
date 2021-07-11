from flask import Flask, request, jsonify
from flask_cors import CORS
import requestService
import getRandomTweet

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})


@app.route('/api', methods=['POST'])
def sentiment_analysis():
    """Get row text as request and returns the given text cleaned along with the decisions about sentiment
    classification from several ml models"""
    classification_dto = requestService.classify_text(request.json['text'])
    return jsonify(classification_dto)


@app.route('/getRandomTweet', methods=['GET'])
def retrieve_random_tweet():
    """Returns one random tweet along with the sentiment label from the testing dataframe"""
    tweet_dto = getRandomTweet.get_tweet()
    return jsonify(tweet_dto.get_response())


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')
