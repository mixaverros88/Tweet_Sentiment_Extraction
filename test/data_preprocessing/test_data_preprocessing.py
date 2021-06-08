from helper_functions.clean_dataset.DataCleaning import DataCleaning
import pandas as pd

# data
text = 'Hello!!!, he said ---and went.   '
data = [[1, text, '', 'positive']]
data_frame = pd.DataFrame(data, columns=['textID', 'text', 'selected_text', 'sentiment'])

sample_cleaning_dataset = DataCleaning(data_frame, 'textID', 'request')
cleaned_sample_data_frame = sample_cleaning_dataset.data_cleaning()

assert cleaned_sample_data_frame.loc[0, 'text'] == 'hello he said and went'
