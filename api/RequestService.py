import pandas as pd
import numpy as np
import pickle

from api.dto.ClassificationDto2 import ClassificationDto2
from helper_functions.clean_dataset.DataCleaning import DataCleaning


class RequestService:

    def __init__(self, text):
        self.text = text

    def convert_target_column(self):
        # assign data of lists.
        data = {'text': [self.text]}

        # Create DataFrame
        dataframe = pd.DataFrame(data)
        sample_cleaning_dataset = DataCleaning(dataframe, 'textID', 'request')
        cleaned_sample_data_frame = sample_cleaning_dataset.data_cleaning()

        print('cleaned request', cleaned_sample_data_frame.iloc[0]['text'])

        bag_of_words = pickle.load(open(
            'C:\\Users\\mverros\\Desktop\\archive\\python_projects\\npl\\Tweet_Sentiment_Extraction\\models\\machine_learning\\serialized\\bag_of_words.sav',
            'rb'))

        vectors = bag_of_words.transform([cleaned_sample_data_frame.iloc[0]['text']])

        print('vectors_bag_of_words', vectors)

        ss = np.zeros(shape=(14559, 1)).reshape(1, -1)
        print(ss)

        # load the model from disk
        loaded_model = pickle.load(open(
            'C:\\Users\\mverros\\Desktop\\archive\\python_projects\\npl\\Tweet_Sentiment_Extraction\\models\\machine_learning\\serialized\\logistic_regression.sav',
            'rb'))
        result = loaded_model.predict_proba(vectors)
        print('result', result)

        dd = ClassificationDto2(cleaned_sample_data_frame, result)
        return dd
