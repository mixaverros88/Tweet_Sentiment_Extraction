from helper_functions.retrieve import dataset as read_dataset
from helper_functions.tokenizer.functions import get_column_values_as_np_array, tokenize_sentence, \
    tokenizing_sentences_and_words
from helper_functions.text_vectorization.Word2VecModel import Word2VecModel
from helper_functions.metrics.ComposeMetrics import ComposeMetrics
from models.machine_learning.LogisticRegressionModel import LogisticRegressionModel
from sklearn.model_selection import train_test_split
import numpy as np
import configparser
config = configparser.RawConfigParser()
config.read('ConfigFile.properties')


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

# Tokenize data frame
corpus = tokenize_sentence(train_data_frame_over_sampling)

# Vectorized - Word2Vec
tokenized_sentences = tokenizing_sentences_and_words(train_data_frame_over_sampling)
word_2_vec = Word2VecModel(tokenized_sentences, config.get('MODELS', 'oversampling.word2vec.word2vec'))
vectors_word_2_vec = word_2_vec.text_vectorization()


def shuffle_along_axis(a, axis):
    idx = np.random.rand(*a.shape).argsort(axis=axis)
    return np.take_along_axis(a, idx, axis=axis)


ss = shuffle_along_axis(target_values, 0)

# Split Train-Test Data
X_train, X_test, y_train, y_test = train_test_split(vectors_word_2_vec.wv.syn0, ss[:len(vectors_word_2_vec.wv.vocab)],
                                                    test_size=0.33, random_state=32)

logistic_regression_model_2 = LogisticRegressionModel(X_train, X_test, y_train, y_test,
                                                      config.get('MODELS', 'oversampling.word2vec.lg'))
logistic_regression_model_results2 = logistic_regression_model_2.results()
# print(logistic_regression_model_results2)

print(y_test)
print(logistic_regression_model_results2)

ComposeMetrics(y_test, logistic_regression_model_results2, 'Logistic Regression Model', [0, 1, 2])
