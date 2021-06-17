import pandas as pd

from api.dto.ClassificationDto import ClassificationDto
from helper_functions.clean_dataset.DataCleaning import DataCleaning
from helper_functions.retrieve.serializedModels import bag_of_words_over_sampling, \
    logistic_regression_over_sampling, svm_over_sampling, nb_over_sampling, \
    multi_layer_perceptron_classifier_over_sampling, decision_tree_over_sampling


class RequestService:

    def __init__(self, requested_text):
        self.requested_text = requested_text

    def classify_text(self):
        # convert text to data frame with one row since the initial implementation of DataCleaning accepts dataframe
        request_text = {'text': [self.requested_text]}
        data_frame = pd.DataFrame(request_text)
        data_cleaning = DataCleaning(data_frame)
        cleaned_data_frame = data_cleaning.data_pre_processing()
        cleaned_text = cleaned_data_frame.iloc[0]['text']
        print('Cleaned Request: ', cleaned_text)

        # Vectorize Cleaned Text With BOW
        bag_of_words_model = bag_of_words_over_sampling()  # Retrieve Model
        bag_of_words_vectors = bag_of_words_model.transform([cleaned_text])

        logistic_regression_model = logistic_regression_over_sampling()  # Retrieve Model
        logistic_regression_probabilities_results = logistic_regression_model.predict_proba(bag_of_words_vectors)

        logistic_regression_results = logistic_regression_model.predict(bag_of_words_vectors)

        svm_model = svm_over_sampling()  # Retrieve Model
        svm_results = svm_model.predict(bag_of_words_vectors)

        nb_model = nb_over_sampling()  # Retrieve Model
        nb_results = nb_model.predict(bag_of_words_vectors.toarray())

        mlp_model = multi_layer_perceptron_classifier_over_sampling()  # Retrieve Model
        mlp_results = mlp_model.predict(bag_of_words_vectors)

        decision_tree_model = decision_tree_over_sampling()  # Retrieve Model
        decision_tree_results = decision_tree_model.predict(bag_of_words_vectors)

        # Vectorize Cleaned Text With WORD2VEC
        # word2vec_model = bag_of_word2vec_sampling()  # Retrieve Model
        # word2vec_vectors = word2vec_model.transform([cleaned_request])
        #
        # logistic_regression_word2vec_model = logistic_regression_word2vec_under_sampling()  # Retrieve Model
        # logistic_regression_word2vec_results = logistic_regression_word2vec_model.predict(word2vec_vectors)

        return ClassificationDto(
            cleaned_data_frame,
            logistic_regression_probabilities_results,
            logistic_regression_results,
            svm_results,
            nb_results,
            mlp_results,
            decision_tree_results
        )
