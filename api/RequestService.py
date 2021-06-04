from helper_functions.clean_dataset.DataCleaning import DataCleaning


class RequestService:

    def __init__(self, text):
        self.text = text
        self.convert_taget_column()

    def convert_taget_column(self):
        # Cleaning Dataset
        sample_cleaning_dataset = DataCleaning(self.text, 'textID')
        cleaned_sample_data_frame = sample_cleaning_dataset.data_cleaning()
        return self.data_frame
