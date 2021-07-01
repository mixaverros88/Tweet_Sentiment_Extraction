from helper.retrieve import dataset as read_dataset
from api.dto.Tweet import Tweet


class GetRandomTweet:

    def get_tweet(self):
        random_row = read_dataset.read_test_data_set().sample()
        for index, row in random_row.iterrows():
            tweet = row['text']
            sentiment = row['sentiment']
        return Tweet(tweet, sentiment)
