from helper.retrieve import dataset as read_dataset
from api.dto.Tweet import Tweet


class GetRandomTweet:

    def get_tweet(self):
        random_tweet = read_dataset.read_test_data_set().sample()
        tweet = random_tweet.iloc[0]['text']
        sentiment = random_tweet.iloc[0]['sentiment']
        return Tweet(tweet, sentiment)
