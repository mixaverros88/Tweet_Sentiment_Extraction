import logging

import pandas as pd


class BalanceDataset:
    def __init__(self, data_frame, target_column):
        self.data_frame = data_frame
        self.target_column = target_column

    def convert_to_balance_dataset(self):
        shuffled_df = self.data_frame.sample(frac=1, random_state=4)  # Shuffle the Dataset.
        logging.info("shuffled_df ", shuffled_df.shape)

        # Put all the 0 class (minority) in a separate dataset.
        negative = shuffled_df.loc[shuffled_df[self.target_column] == 0]

        # Randomly select 4985 observations from the 1 (majority class)
        neutral = shuffled_df.loc[shuffled_df[self.target_column] == 1].sample(n=7781, random_state=42)
        positive = shuffled_df.loc[shuffled_df[self.target_column] == 2].sample(n=7781, random_state=42)

        # Concatenate both dataframes again
        normalized_df = pd.concat([negative, neutral, positive])
        return normalized_df
