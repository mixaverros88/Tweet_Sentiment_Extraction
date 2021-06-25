from helper_functions.retrieve import dataset as read_dataset
from helper_functions.tokenizer.functions import get_column_values_as_np_array, tokenize_sentence, \
    tokenizing_sentences_and_words
from helper_functions.text_vectorization.Word2VecModel import Word2VecModel
from helper_functions.metrics.ComposeMetrics import ComposeMetrics
from models.machine_learning.LogisticRegressionModel import LogisticRegressionModel
from models.machine_learning.SvmModel import SvmModel
from models.machine_learning.KNeighborsModel import KNeighborsModel
from models.machine_learning.DecisionTreeModel import DecisionTreeModel
from models.machine_learning.GaussianNBModel import GaussianNBModel
from models.neural.MLPClassifierModel import MLPClassifierModel
from sklearn.model_selection import train_test_split
import numpy as np
import configparser

config = configparser.RawConfigParser()
config.read('ConfigFile.properties')
data_set = config.get('STR', 'data.over.sampling')
word_embedding = config.get('STR', 'word.embedding.word2vec')
target_column = config.get('STR', 'target.column')

# Retrieve Data Frames
train_data_frame_over_sampling = read_dataset.read_cleaned_train_data_set_over_sampling()
test_data_frame = read_dataset.read_cleaned_test_data_set()

# TODO: check why nulls rows
# Remove Null rows
train_data_frame_over_sampling.dropna(inplace=True)
test_data_frame.dropna(inplace=True)

# Get Target Values as Numpy Array
target_values = get_column_values_as_np_array(target_column, train_data_frame_over_sampling)

df = train_data_frame_over_sampling
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

# Tokenize data frame
corpus = tokenize_sentence(train_data_frame_over_sampling)

# Vectorized - Word2Vec
tokenized_sentences = tokenizing_sentences_and_words(train_data_frame_over_sampling)
word_2_vec = Word2VecModel(tokenized_sentences, config.get('MODELS', 'oversampling.word2vec.word2vec'))
word2vec_model = word_2_vec.text_vectorization()

from nltk import sent_tokenize, word_tokenize
from collections import Counter


def get_vec2(x):
    word_tokens = word_tokenize(x)
    np_vec = []
    for word in word_tokens:
        vec = word2vec_model.wv[str(word)]
        np_vec.append(vec)
    aa = np.average(np_vec, axis=0)
    print(type((np_vec)))
    print(type((aa)))
    nn = np.array(np_vec)
    print(type((nn)))
    return aa


df['vec2'] = df['text'].apply(lambda x: get_vec2(x))

print(df['vec2'].iloc[0])
X = df['vec2'].to_numpy()
print(X)
X = X.reshape(-1, 1)
print(X)
X = np.concatenate(np.concatenate(X, axis=0), axis=0).reshape(-1, 100)
print(X)

y = target_values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

print(df.head())

logistic_regression_model = LogisticRegressionModel(X_train, X_test, y_train, y_test,
                                                    config.get('MODELS', 'oversampling.word2vec.lg'))
logistic_regression_model = logistic_regression_model.results()

ComposeMetrics(logistic_regression_model.score, y_test, logistic_regression_model.prediction,
               'Logistic Regression Model', data_set, word_embedding)

# Support Vector Machine
svm_model = SvmModel(X_train, X_test, y_train, y_test, config.get('MODELS', 'oversampling.word2vec.svm'))
svm_y_predict = svm_model.results()

ComposeMetrics(svm_y_predict.score, y_test, svm_y_predict.prediction, config.get('MODELNAME', 'model.svm'), data_set,
               word_embedding)

# Gaussian Naive Bayes
nb_model = GaussianNBModel(X_train, X_test, y_train, y_test, config.get('MODELS', 'oversampling.word2vec.gaussian'), 'sss')
nb_y_predict = nb_model.results()

ComposeMetrics(nb_y_predict.score, y_test, nb_y_predict.prediction, config.get('MODELNAME', 'model.nb'), data_set,
               word_embedding)

# MLP Classifier
neural_network = MLPClassifierModel(X_train, X_test, y_train, y_test, config.get('MODELS', 'oversampling.word2vec.mlp'), 'sss')
neural_network_predict = neural_network.results()

ComposeMetrics(neural_network_predict.score, y_test, neural_network_predict.prediction,
               config.get('MODELNAME', 'model.mlp'), data_set, word_embedding)

decision_tree = DecisionTreeModel(X_train, X_test, y_train, y_test, config.get('MODELS', 'oversampling.word2vec.dt'))
decision_tree_predict = decision_tree.results()

# ComposeMetrics( decision_tree_predict.score, y_test, decision_tree_predict.prediction, config.get('MODELNAME', 'model.dt'), data_set, word_embedding)

# K Neighbors
kneighbors_model = KNeighborsModel(X_train, X_test, y_train, y_test,
                                   config.get('MODELS', 'oversampling.word2vec.k_neighbors'))
kneighbors_model_predict = kneighbors_model.results()

# ComposeMetrics(, kneighbors_model.score, y_test, kneighbors_model.prediction, config.get('MODELNAME', 'model.kn'), data_set, word_embedding)
