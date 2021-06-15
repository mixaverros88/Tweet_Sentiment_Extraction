from helper_functions.retrieve import dataset as read_dataset
from helper_functions.tokenizer.functions import get_column_values_as_np_array, tokenize_sentence
from helper_functions.text_vectorization.BoW import BoW
from helper_functions.text_vectorization.Word2VecModel import Word2VecModel
from helper_functions.metrics.ComposeMetrics import ComposeMetrics
from models.machine_learning.LogisticRegressionModel import LogisticRegressionModel
from models.machine_learning.SvmModel import SvmModel
from models.machine_learning.GaussianNBModel import GaussianNBModel
from models.neural.MLPClassifierModel import MLPClassifierModel
from models.machine_learning.KNeighborsModel import KNeighborsModel
from sklearn.model_selection import train_test_split
import numpy as np
import nltk

target_column = 'sentiment'

# Retrieve Data Frames
train_data_frame_under_sampling = read_dataset.read_cleaned_train_data_set_under_sampling()
test_data_frame = read_dataset.read_cleaned_test_data_set()

# TODO: check why nulls rows
# Remove Null rows
train_data_frame_under_sampling.dropna(inplace=True)
test_data_frame.dropna(inplace=True)

# Get Target Values as Numpy Array
train_target_values_under_sampling = get_column_values_as_np_array(target_column, train_data_frame_under_sampling)
test_target_values = get_column_values_as_np_array(target_column, test_data_frame)

target_values = np.concatenate((train_target_values_under_sampling, test_target_values))

# Tokenize data frame
train_corpus_under_sampling = tokenize_sentence(train_data_frame_under_sampling)
test_corpus = tokenize_sentence(test_data_frame)

corpus = train_corpus_under_sampling + test_corpus

# Vectorized - BOW
bag_of_words_under_sampling = BoW(corpus)
vectors_bag_of_words_under_sampling = bag_of_words_under_sampling.vectorize_text()

# Split Train-Test Data
X_train, X_test, y_train, y_test = train_test_split(vectors_bag_of_words_under_sampling, target_values, test_size=0.3,
                                                    random_state=32)

# Logistic Regression
logistic_regression_model = LogisticRegressionModel(X_train, X_test, y_train, y_test,
                                                    'logistic_regression_under_sampling')
logistic_regression_y_predict = logistic_regression_model.results()

ComposeMetrics(y_test, logistic_regression_y_predict, 'Logistic Regression Model', [0, 1, 2])

# Support Vector Machine
svm_model = SvmModel(X_train, X_test, y_train, y_test, 'svm_over_sampling')
svm_y_predict = svm_model.results()

ComposeMetrics(y_test, svm_y_predict, 'SVM Model', [0, 1, 2])

# Gaussian Naive Bayes
nb_model = GaussianNBModel(X_train, X_test, y_train, y_test, 'gaussian_over_sampling')
nb_y_predict = nb_model.results()

ComposeMetrics(y_test, nb_y_predict, 'NB Model', [0, 1, 2])

# MLP Classifier
neural_network = MLPClassifierModel(X_train, X_test, y_train, y_test, 'mlp_over_sampling')
neural_network_predict = neural_network.results()

ComposeMetrics(y_test, neural_network_predict, 'MLPClassifier Model', [0, 1, 2])

# K Neighbors

kneighbors_model = KNeighborsModel(X_train, X_test, y_train, y_test, 'kneighbors_over_sampling')
kneighbors_model_predict = kneighbors_model.results()

# ComposeMetrics(y_test, kneighbors_model, 'KNeighbors Classifier', [0, 1, 2])


# Vectorized - Word2Vec
text_title_array = train_data_frame_under_sampling["text"].values
tokenized_sentences = [nltk.word_tokenize(title) for title in text_title_array]
word_2_vec = Word2VecModel(tokenized_sentences)
vectors_word_2_vec = word_2_vec.vectorize_text()

# X_train, X_test, y_train, y_test = train_test_split(vectors_word_2_vec, target_values, test_size=0.3, random_state=109)
#
# logistic_regression_model2 = LogisticRegressionModel(X_train, X_test, y_train, y_test)
# logistic_regression_model_results2 = logistic_regression_model2.results()
# print(logistic_regression_model_results2)
