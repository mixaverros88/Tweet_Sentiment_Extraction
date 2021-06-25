import pandas as pd

from api.dto.ClassificationDto import ClassificationDto
from helper_functions.clean_dataset.DataCleaning import DataCleaning
from helper_functions.retrieve.serializedModels import bag_of_words_over_sampling, \
    bag_of_words_logistic_regression_over_sampling, bag_of_words_svm_over_sampling, bag_of_words_nb_over_sampling, \
    bag_of_words_multi_layer_perceptron_classifier_over_sampling, bag_of_words_decision_tree_over_sampling, \
    tfidf_over_sampling, tfidf_logistic_regression_over_sampling, tfidf_svm_over_sampling, tfidf_nb_over_sampling, \
    tfidf_multi_layer_perceptron_classifier_over_sampling, tfidf_decision_tree_over_sampling, word2vec_over_sampling, \
    word2vec_logistic_regression_over_sampling, word2vec_svm_over_sampling, \
    word2vec_multi_layer_perceptron_classifier_over_sampling, word2vec_decision_tree_over_sampling


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

        # BOW
        bag_of_words_model = bag_of_words_over_sampling()  # Retrieve Model
        bag_of_words_vectors = bag_of_words_model.transform([cleaned_text])

        bag_of_words_logistic_regression_model = bag_of_words_logistic_regression_over_sampling()  # Retrieve Model
        bag_of_words_logistic_regression_probabilities_results = bag_of_words_logistic_regression_model.predict_proba(
            bag_of_words_vectors)
        bag_of_words_logistic_regression_results = bag_of_words_logistic_regression_model.predict(bag_of_words_vectors)

        bag_of_words_svm_model = bag_of_words_svm_over_sampling()  # Retrieve Model
        bag_of_words_svm_results = bag_of_words_svm_model.predict(bag_of_words_vectors)

        bag_of_words_nb_model = bag_of_words_nb_over_sampling()  # Retrieve Model
        bag_of_words_nb_results = bag_of_words_nb_model.predict(bag_of_words_vectors.toarray())

        bag_of_words_mlp_model = bag_of_words_multi_layer_perceptron_classifier_over_sampling()  # Retrieve Model
        bag_of_words_mlp_results = bag_of_words_mlp_model.predict(bag_of_words_vectors)

        bag_of_words_decision_tree_model = bag_of_words_decision_tree_over_sampling()  # Retrieve Model
        bag_of_words_decision_tree_results = bag_of_words_decision_tree_model.predict(bag_of_words_vectors)

        # Word2Vec
        word2vec_model = word2vec_over_sampling()  # Retrieve Model
        from nltk import sent_tokenize, word_tokenize
        import numpy as np
        word_tokens = word_tokenize(cleaned_text)
        np_vec = []
        for word in word_tokens:
            vec = word2vec_model.wv[str(word)]
            np_vec.append(vec)
        aa = np.average(np_vec, axis=0)
        print(type((np_vec)))
        print(type((aa)))
        nn = np.array(np_vec)
        print(type((nn)))

        word2vec_logistic_regression_model = word2vec_logistic_regression_over_sampling()  # Retrieve Model
        word2vec_logistic_regression_probabilities_results = word2vec_logistic_regression_model.predict_proba(np_vec)
        word2vec_logistic_regression_model_results = word2vec_logistic_regression_model.predict(np_vec)

        word2vec_svm_model = word2vec_svm_over_sampling()  # Retrieve Model
        word2vec_svm_results = word2vec_svm_model.predict(np_vec)

        # word2vec_nb_model = word2vec_nb_over_sampling()  # Retrieve Model
        # word2vec_nb_results = word2vec_nb_model.predict(np_vec.toarray())

        word2vec_mlp_model = word2vec_multi_layer_perceptron_classifier_over_sampling()  # Retrieve Model
        word2vec_mlp_results = word2vec_mlp_model.predict(np_vec)

        word2vec_decision_tree_model = word2vec_decision_tree_over_sampling()  # Retrieve Model
        word2vec_decision_tree_results = word2vec_decision_tree_model.predict(np_vec)

        # Tfidf
        tfidf_model = tfidf_over_sampling()
        tfidf_model_vectors = tfidf_model.transform([cleaned_text])

        tfidf_logistic_regression_model = tfidf_logistic_regression_over_sampling()  # Retrieve Model
        tfidf_logistic_regression_probabilities_results = tfidf_logistic_regression_model.predict_proba(
            tfidf_model_vectors)
        tfidf_logistic_regression_results = tfidf_logistic_regression_model.predict(tfidf_model_vectors)

        tfidf_svm_model = tfidf_svm_over_sampling()  # Retrieve Model
        tfidf_svm_results = tfidf_svm_model.predict(tfidf_model_vectors)

        tfidf_words_nb_model = tfidf_nb_over_sampling()  # Retrieve Model
        tfidf_nb_results = tfidf_words_nb_model.predict(tfidf_model_vectors.toarray())

        tfidf_mlp_model = tfidf_multi_layer_perceptron_classifier_over_sampling()  # Retrieve Model
        tfidf_mlp_results = tfidf_mlp_model.predict(tfidf_model_vectors)

        tfidf_decision_tree_model = tfidf_decision_tree_over_sampling()  # Retrieve Model
        tfidf_decision_tree_results = tfidf_decision_tree_model.predict(tfidf_model_vectors)

        return ClassificationDto(
            cleaned_data_frame,
            bag_of_words_logistic_regression_probabilities_results,
            bag_of_words_logistic_regression_results,
            bag_of_words_svm_results,
            bag_of_words_nb_results,
            bag_of_words_mlp_results,
            bag_of_words_decision_tree_results,
            tfidf_logistic_regression_probabilities_results,
            tfidf_logistic_regression_results,
            tfidf_svm_results,
            tfidf_nb_results,
            tfidf_mlp_results,
            tfidf_decision_tree_results,
            word2vec_logistic_regression_model_results,
            word2vec_logistic_regression_probabilities_results,
            word2vec_svm_results,
            word2vec_mlp_results,
            word2vec_decision_tree_results
        )
