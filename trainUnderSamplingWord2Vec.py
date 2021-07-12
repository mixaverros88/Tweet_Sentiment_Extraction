import utils.dataset as data
from models.text_vectorization.Word2VecModel import Word2VecModel
from utils.ComposeMetrics import ComposeMetrics
from models.machine_learning.LogisticRegressionModel import LogisticRegressionModel
from models.machine_learning.SvmModel import SvmModel
from models.machine_learning.KNeighborsModel import KNeighborsModel
from models.machine_learning.DecisionTreeModel import DecisionTreeModel
from models.neural.MLPClassifierModel import MLPClassifierModel
from sklearn.model_selection import train_test_split
from utils.functions import tokenizing_sentences_and_words_data_frame, get_column_values_as_np_array, tokenize_sentence, \
    count_word_occurrences, remove_words_from_corpus, count_the_most_common_words_in_data_set_convert, \
    count_the_most_common_words_in_data_set, convert_data_frame_sentence_to_vector_array
import configparser

config = configparser.RawConfigParser()
config.read('ConfigFile.properties')
data_set = config.get('STR', 'data.under.sampling')
word_embedding = config.get('STR', 'word.embedding.word2vec')
target_column = config.get('STR', 'target.column')
test_size = float(config.get('PROJECT', 'test.size'))
random_state = int(config.get('PROJECT', 'random.state'))
remove_words_by_occur_size = int(config.get('PROJECT', 'remove.words.occur.size'))
remove_most_common_word_size = int(config.get('PROJECT', 'remove.most.common.word'))

# Retrieve Data Frames
train_data_frame_under_sampling = data.read_cleaned_train_data_set_under_sampling()

# Remove Null rows
train_data_frame_under_sampling.dropna(inplace=True)

# Get Target Values as Numpy Array
target_values = get_column_values_as_np_array(target_column, train_data_frame_under_sampling)

# List of words that occurs 3 or less times
list_of_words_tha_occurs_3_or_less_times = count_word_occurrences(train_data_frame_under_sampling,
                                                                  remove_words_by_occur_size)

# List of top 15 most common word
most_common_words = count_the_most_common_words_in_data_set(train_data_frame_under_sampling, 'text',
                                                            remove_most_common_word_size)
most_common_words = count_the_most_common_words_in_data_set_convert(most_common_words)

# Tokenize data frame
corpus = tokenize_sentence(train_data_frame_under_sampling)

# Remove from corpus the given list of words
corpus = remove_words_from_corpus(corpus, list_of_words_tha_occurs_3_or_less_times + most_common_words)

# Vectorized - Word2Vec
tokenized_sentences = tokenizing_sentences_and_words_data_frame(train_data_frame_under_sampling)
word_2_vec = Word2VecModel(tokenized_sentences, config.get('MODELS', 'under_sampling.word2vec.word2vec'))
word2vec_model = word_2_vec.text_vectorization()

X = convert_data_frame_sentence_to_vector_array(word2vec_model, train_data_frame_under_sampling)

# Split Train-Test Data
X_train, X_test, y_train, y_test = \
    train_test_split(X, target_values, test_size=test_size, random_state=random_state, stratify=target_values)

# Logistic Regression
logistic_regression_params = {'C': 121, 'penalty': 'l2', 'max_iter': 1000, 'solver': 'liblinear'}
logistic_regression_model = LogisticRegressionModel(X_train, X_test, y_train, y_test,
                                                    config.get('MODELS', 'under_sampling.word2vec.lg'),
                                                    logistic_regression_params)
logistic_regression_model = logistic_regression_model.results()

ComposeMetrics(
    logistic_regression_model.score,
    y_test,
    logistic_regression_model.prediction,
    config.get('MODELNAME', 'model.lg'),
    data_set,
    word_embedding)

# Support Vector Machine
svm_params = {'kernel': 'linear'}
svm_model = SvmModel(X_train, X_test, y_train, y_test, config.get('MODELS', 'under_sampling.word2vec.svm'), svm_params)
svm_y_predict = svm_model.results()

ComposeMetrics(
    svm_y_predict.score,
    y_test,
    svm_y_predict.prediction,
    config.get('MODELNAME', 'model.svm'),
    data_set,
    word_embedding)

# Gaussian Naive Bayes
# nb_params = {'alpha': 1.5}
# nb_model = GaussianNBModel(X_train, X_test, y_train, y_test, config.get('MODELS', 'under_sampling.word2vec.gaussian'), 'sss')
# nb_y_predict = nb_model.results()
#
# ComposeMetrics(nb_y_predict.score, y_test, nb_y_predict.prediction, config.get('MODELNAME', 'model.nb'), data_set,
#                word_embedding)

# MLP Classifier
neural_network_params = {'activation': 'tanh', 'alpha': 0.05, 'hidden_layer_sizes': (5, 5, 5),
                         'learning_rate': 'adaptive', 'max_iter': 1000}
neural_network = MLPClassifierModel(X_train, X_test, y_train, y_test,
                                    config.get('MODELS', 'under_sampling.word2vec.mlp'),
                                    'sss', neural_network_params)
neural_network_predict = neural_network.results()

ComposeMetrics(
    neural_network_predict.score,
    y_test,
    neural_network_predict.prediction,
    config.get('MODELNAME', 'model.mlp'),
    data_set,
    word_embedding)

# Decision Tree
decision_tree_params = {'max_depth': 5, 'max_leaf_nodes': 18, 'min_samples_split': 3}
decision_tree = DecisionTreeModel(X_train, X_test, y_train, y_test, config.get('MODELS', 'under_sampling.word2vec.dt'),
                                  decision_tree_params)
decision_tree_predict = decision_tree.results()

ComposeMetrics(
    decision_tree_predict.score,
    y_test,
    decision_tree_predict.prediction,
    config.get('MODELNAME', 'model.dt'),
    data_set,
    word_embedding)

# K Neighbors
k_neighbors_params = {'metric': 'euclidean', 'weights': 'distance'}
k_neighbors_model = KNeighborsModel(X_train, X_test, y_train, y_test,
                                    config.get('MODELS', 'under_sampling.word2vec.k_neighbors'), k_neighbors_params)
k_neighbors_model_predict = k_neighbors_model.results()

ComposeMetrics(
    k_neighbors_model_predict.score,
    y_test,
    k_neighbors_model_predict.prediction,
    config.get('MODELNAME', 'model.kn'),
    data_set,
    word_embedding)

# import numpy as np
# import pandas as pd
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report
# import spacy
# import en_core_web_lg
#
# nlp = en_core_web_lg.load()
#
#
#
#
# def get_vec(x):
#     doc = nlp(str(x))
#     vec = doc.vector
#     return vec
#
#
# df['vec'] = df['text'].apply(lambda x: get_vec(x))
#
# print(df.head())
# X = df['vec'].to_numpy()
# X = X.reshape(-1, 1)
# X = np.concatenate(np.concatenate(X, axis=0), axis=0).reshape(-1, 300)
#
# y = target_values
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
#
# print(X_train.shape)
# print(X_test.shape)
#
# clf = LogisticRegression(solver='liblinear')
# clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)
# print(classification_report(y_test, y_pred))
