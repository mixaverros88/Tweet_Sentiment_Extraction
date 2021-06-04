import pandas as pd

from helper_functions.clean_dataset.DataCleaning import DataCleaning


class RequestService:

    def __init__(self, text):
        self.text = text

    def convert_target_column(self):
        # assign data of lists.
        data = {'text': [self.text]}

        # Create DataFrame
        dataframe = pd.DataFrame(data)
        sample_cleaning_dataset = DataCleaning(dataframe, 'textID')
        cleaned_sample_data_frame = sample_cleaning_dataset.data_cleaning()
        print('---' + cleaned_sample_data_frame)
        return cleaned_sample_data_frame
