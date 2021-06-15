import os
import pickle


def bag_of_words():
    return pickle.load(
        open(os.path.abspath(__file__ + "/../../../serializedModels/bag_of_words.sav"), 'rb'))


def bag_of_words_over_sampling():
    return pickle.load(
        open(os.path.abspath(__file__ + "/../../../serializedModels/bag_of_words_over_sampling.sav"), 'rb'))


def logistic_regression():
    return pickle.load(
        open(os.path.abspath(__file__ + "/../../../serializedModels/logistic_regression.sav"), 'rb'))


def logistic_regression_over_sampling():
    return pickle.load(
        open(os.path.abspath(__file__ + "/../../../serializedModels/logistic_regression_over_sampling.sav"), 'rb'))


def svm():
    return pickle.load(
        open(os.path.abspath(__file__ + "/../../../serializedModels/svm.sav"), 'rb'))


def svm_over_sampling():
    return pickle.load(
        open(os.path.abspath(__file__ + "/../../../serializedModels/mlp_over_sampling.sav"), 'rb'))


def MLPClassifier():
    return pickle.load(
        open(os.path.abspath(__file__ + "/../../../serializedModels/MLPClassifierModel.sav"), 'rb'))


def MLPClassifier_over_sampling():
    return pickle.load(
        open(os.path.abspath(__file__ + "/../../../serializedModels/mlp_over_sampling.sav"), 'rb'))


def nb():
    return pickle.load(
        open(os.path.abspath(__file__ + "/../../../serializedModels/gaussian_nb.sav"), 'rb'))


def nb_over_sampling():
    return pickle.load(
        open(os.path.abspath(__file__ + "/../../../serializedModels/gaussian_over_sampling.sav"), 'rb'))