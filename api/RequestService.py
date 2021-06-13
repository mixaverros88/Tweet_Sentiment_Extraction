import pandas as pd

from api.dto.ClassificationDto import ClassificationDto
from helper_functions.clean_dataset.DataCleaning import DataCleaning
from helper_functions.retrieve.serializedModels import bag_of_words, logistic_regression, svm, nb, MLPClassifier


class RequestService:

    def __init__(self, text):
        self.text = text

    def classify_text(self):
        request_text = {'text': [self.text]}
        data_frame = pd.DataFrame(request_text)
        data_cleaning = DataCleaning(data_frame, 'textID', 'request')
        cleaned_data_frame = data_cleaning.data_cleaning()
        cleaned_request = cleaned_data_frame.iloc[0]['text']
        print('Cleaned Request: ', cleaned_request)

        bag_of_words_model = bag_of_words()  # Retrieve Model
        vectors = bag_of_words_model.transform([cleaned_request])

        logistic_regression_model = logistic_regression()  # Retrieve Model
        logistic_regression_proba_results = logistic_regression_model.predict_proba(vectors)

        logistic_regression_results = logistic_regression_model.predict(vectors)

        svm_model = svm()  # Retrieve Model
        svm_results = svm_model.predict(vectors)

        nb_model = nb()  # Retrieve Model
        nb_results = nb_model.predict(vectors.toarray())

        mlp_model = MLPClassifier()
        mlp_results = mlp_model.predict(vectors)

        return ClassificationDto(
            cleaned_data_frame,
            logistic_regression_proba_results,
            logistic_regression_results,
            svm_results,
            nb_results,
            mlp_results
        )
