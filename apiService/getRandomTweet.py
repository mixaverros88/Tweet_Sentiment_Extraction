from utils.retrieve.dataset import read_test_data_set
from TweetDto import TweetDto


def get_tweet():
    """Retrieve random tweet from test dataset """
    random_tweet = read_test_data_set().sample()
    tweet = random_tweet.iloc[0]['text']
    sentiment = random_tweet.iloc[0]['sentiment']
    return TweetDto(tweet, sentiment)
