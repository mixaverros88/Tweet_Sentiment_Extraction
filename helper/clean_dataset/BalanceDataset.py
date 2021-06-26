import logging
from imblearn.over_sampling import RandomOverSampler
import pandas as pd
import numpy as np


class BalanceDataset:
    def __init__(self, data_frame, target_column):
        self.data_frame = data_frame
        self.target_column = target_column

    def under_sampling_majority_class(self):
        """Perform Under Sampling"""
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

    def over_sampling_majority_class(self):
        """Perform Over Sampling"""
        # https://www.programcreek.com/python/example/123411/imblearn.over_sampling.RandomOverSampler
        # Extract predicted column
        y = np.squeeze(self.data_frame[[self.target_column]])

        # Copy the dataframe without the predicted column
        temp_dataframe = self.data_frame.drop([self.target_column], axis=1)

        # Initialize and fit the under sampler
        over_sampler = RandomOverSampler(random_state=32)
        x_over_sampled, y_over_sampled = over_sampler.fit_resample(temp_dataframe, y)

        # Build the resulting under sampled dataframe
        result = pd.DataFrame(x_over_sampled)

        # Restore the column names
        result.columns = temp_dataframe.columns

        # Restore the y values
        y_over_sampled = pd.Series(y_over_sampled)
        result[self.target_column] = y_over_sampled

        return result
