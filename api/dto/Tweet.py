def map_sentiment(sentiment):
    if sentiment == 0:
        return 'Negative'
    if sentiment == 1:
        return 'Neutral'
    if sentiment == 2:
        return 'Positive'
    # TODO

class Tweet:

    def __init__(self, tweet, sentiment):
        self.tweet = tweet
        self.sentiment = sentiment

    def get_response(self):
        return {
            'tweet': str(self.tweet),
            'sentiment': str(self.sentiment)
        }
