from utils.retrieve.dataset import read_test_data_set
from Tweet import Tweet


def get_tweet():
    random_tweet = read_test_data_set().sample()
    tweet = random_tweet.iloc[0]['text']
    sentiment = random_tweet.iloc[0]['sentiment']
    return Tweet(tweet, sentiment)
