"""Tweet DTO"""


class TweetDto:

    def __init__(self, tweet, sentiment):
        self.tweet = tweet
        self.sentiment = sentiment

    def get_response(self):
        return {
            'tweet': str(self.tweet),
            'sentiment': str(self.sentiment)
        }
