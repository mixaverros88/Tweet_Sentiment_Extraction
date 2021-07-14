import pickle
from definitions import ROOT_DIR

""" Retrieve Trained Models """

print(__name__)


def bag_of_words_over_sampling():
    return pickle.load(open(ROOT_DIR + "/apiService/serializedModels/bag_of_words_over_sampling.sav", 'rb'))


def bag_of_words_under_sampling():
    return pickle.load(open(ROOT_DIR + "/apiService/serializedModels/bag_of_words_under_sampling.sav", 'rb'))


def word2vec_over_sampling():
    return pickle.load(open(ROOT_DIR + "/apiService/serializedModels/word2vec_over_sampling.sav", 'rb'))


def word2vec_under_sampling():
    return pickle.load(open(ROOT_DIR + "/apiService/serializedModels/word2vec_under_sampling.sav", 'rb'))


def tfidf_over_sampling():
    return pickle.load(open(ROOT_DIR + "/apiService/serializedModels/Tfidf_over_sampling.sav", 'rb'))


def tfidf_under_sampling():
    return pickle.load(open(ROOT_DIR + "/apiService/serializedModels/Tfidf_under_sampling.sav", 'rb'))


def bag_of_words_logistic_regression_under_sampling():
    return pickle.load(
        open(ROOT_DIR + "/apiService/serializedModels/bag_of_words_logistic_regression_under_sampling.sav", 'rb'))


def bag_of_words_logistic_regression_over_sampling():
    return pickle.load(
        open(ROOT_DIR + "/apiService/serializedModels/bag_of_words_logistic_regression_over_sampling.sav", 'rb'))


def bag_of_words_k_neighbors_under_sampling():
    return pickle.load(
        open(ROOT_DIR + "/apiService/serializedModels/bag_of_words_k_neighbors_under_sampling.sav", 'rb'))


def bag_of_words_k_neighbors_over_sampling():
    return pickle.load(
        open(ROOT_DIR + "/apiService/serializedModels/bag_of_words_k_neighbors_over_sampling.sav", 'rb'))


def word2vec_k_neighbors_over_sampling():
    return pickle.load(
        open(ROOT_DIR + "/apiService/serializedModels/word2vec_k_neighbors_over_sampling.sav", 'rb'))


def word2vec_k_neighbors_under_sampling():
    return pickle.load(
        open(ROOT_DIR + "/apiService/serializedModels/word2vec_k_neighbors_under_sampling.sav", 'rb'))


def tfidf_k_neighbors_over_sampling():
    return pickle.load(
        open(ROOT_DIR + "/apiService/serializedModels/tfidf_k_neighbors_over_sampling.sav", 'rb'))


def tfidf_k_neighbors_under_sampling():
    return pickle.load(
        open(ROOT_DIR + "/apiService/serializedModels/tfidf_k_neighbors_under_sampling.sav", 'rb'))


def word2vec_logistic_regression_over_sampling():
    return pickle.load(
        open(ROOT_DIR + "/apiService/serializedModels/word2vec_logistic_regression_over_sampling.sav", 'rb'))


def word2vec_logistic_regression_under_sampling():
    return pickle.load(
        open(ROOT_DIR + "/apiService/serializedModels/word2vec_logistic_regression_under_sampling.sav", 'rb'))


def tfidf_logistic_regression_over_sampling():
    return pickle.load(
        open(ROOT_DIR + "/apiService/serializedModels/Tfidf_logistic_regression_over_sampling.sav", 'rb'))


def tfidf_logistic_regression_under_sampling():
    return pickle.load(
        open(ROOT_DIR + "/apiService/serializedModels/Tfidf_logistic_regression_under_sampling.sav", 'rb'))


def bag_of_words_svm_under_sampling():
    return pickle.load(open(ROOT_DIR + "/apiService/serializedModels/bag_of_words_svm_under_sampling.sav", 'rb'))


def bag_of_words_svm_over_sampling():
    return pickle.load(open(ROOT_DIR + "/apiService/serializedModels/bag_of_words_svm_over_sampling.sav", 'rb'))


def word2vec_svm_over_sampling():
    return pickle.load(open(ROOT_DIR + "/apiService/serializedModels/word2vec_svm_over_sampling.sav", 'rb'))


def word2vec_svm_under_sampling():
    return pickle.load(open(ROOT_DIR + "/apiService/serializedModels/word2vec_svm_under_sampling.sav", 'rb'))


def tfidf_svm_over_sampling():
    return pickle.load(open(ROOT_DIR + "/apiService/serializedModels/Tfidf_svm_over_sampling.sav", 'rb'))


def tfidf_svm_under_sampling():
    return pickle.load(open(ROOT_DIR + "/apiService/serializedModels/Tfidf_svm_under_sampling.sav", 'rb'))


def bag_of_words_multi_layer_perceptron_classifier_over_sampling():
    return pickle.load(open(ROOT_DIR + "/apiService/serializedModels/bag_of_words_mlp_over_sampling.sav", 'rb'))


def tfidf_multi_layer_perceptron_classifier_over_sampling():
    return pickle.load(open(ROOT_DIR + "/apiService/serializedModels/Tfidf_mlp_over_sampling.sav", 'rb'))


def tfidf_multi_layer_perceptron_classifier_under_sampling():
    return pickle.load(open(ROOT_DIR + "/apiService/serializedModels/Tfidf_mlp_under_sampling.sav", 'rb'))


def word2vec_multi_layer_perceptron_classifier_over_sampling():
    return pickle.load(open(ROOT_DIR + "/apiService/serializedModels/word2vec_mlp_over_sampling.sav", 'rb'))


def word2vec_multi_layer_perceptron_classifier_under_sampling():
    return pickle.load(open(ROOT_DIR + "/apiService/serializedModels/word2vec_mlp_under_sampling.sav", 'rb'))


def bag_of_words_multi_layer_perceptron_classifier_under_sampling():
    return pickle.load(open(ROOT_DIR + "/apiService/serializedModels/bag_of_words_mlp_under_sampling.sav", 'rb'))


def bag_of_words_nb_under_sampling():
    return pickle.load(open(ROOT_DIR + "/apiService/serializedModels/bag_of_words_gaussian_under_sampling.sav", 'rb'))


def bag_of_words_nb_over_sampling():
    return pickle.load(open(ROOT_DIR + "/apiService/serializedModels/bag_of_words_gaussian_over_sampling.sav", 'rb'))


def tfidf_nb_over_sampling():
    return pickle.load(open(ROOT_DIR + "/apiService/serializedModels/Tfidf_gaussian_over_sampling.sav", 'rb'))


def tfidf_nb_under_sampling():
    return pickle.load(open(ROOT_DIR + "/apiService/serializedModels/Tfidf_gaussian_under_sampling.sav", 'rb'))


def bag_of_words_decision_tree_over_sampling():
    return pickle.load(
        open(ROOT_DIR + "/apiService/serializedModels/bag_of_words_decision_tree_over_sampling.sav", 'rb'))


def word2vec_decision_tree_over_sampling():
    return pickle.load(open(ROOT_DIR + "/apiService/serializedModels/word2vec_decision_tree_over_sampling.sav", 'rb'))


def word2vec_decision_tree_under_sampling():
    return pickle.load(open(ROOT_DIR + "/apiService/serializedModels/word2vec_decision_tree_under_sampling.sav", 'rb'))


def tfidf_decision_tree_over_sampling():
    return pickle.load(open(ROOT_DIR + "/apiService/serializedModels/Tfidf_decision_tree_over_sampling.sav", 'rb'))


def tfidf_decision_tree_under_sampling():
    return pickle.load(open(ROOT_DIR + "/apiService/serializedModels/Tfidf_decision_tree_under_sampling.sav", 'rb'))


def bag_of_words_decision_tree_under_sampling():
    return pickle.load(
        open(ROOT_DIR + "/apiService/serializedModels/bag_of_words_decision_tree_under_sampling.sav", 'rb'))


def main():
    pass


if __name__ == '__main__':
    # execute only if run as the entry point into the program
    main()
