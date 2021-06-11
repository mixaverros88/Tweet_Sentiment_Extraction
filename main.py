from helper_functions.read_dataset import functions as read_dataset
from helper_functions.convert_test.functions import get_column_values_as_np_array, tokenize_sentence, get_corpus
from helper_functions.text_vectorization.BoW import BoW
from helper_functions.text_vectorization.Word2VecModel import Word2VecModel
from helper_functions.metrics.ComposeMetrics import ComposeMetrics
from models.machine_learning.logisticregressionmodel import LogisticRegressionModel
from sklearn.model_selection import train_test_split
import numpy as np
import nltk

target_column = 'sentiment'

# Retrieve Data Frames
sample_data_frame = read_dataset.read_cleaned_sample_data_set()
train_data_frame = read_dataset.read_cleaned_train_data_set()
test_data_frame = read_dataset.read_cleaned_test_data_set()

# TODO: check why nulls rows
# Remove Null rows
sample_data_frame.dropna(inplace=True)
train_data_frame.dropna(inplace=True)
test_data_frame.dropna(inplace=True)

# Get Target Values as Numpy Array
sample_target_values = get_column_values_as_np_array(target_column, sample_data_frame)
train_target_values = get_column_values_as_np_array(target_column, train_data_frame)
test_target_values = get_column_values_as_np_array(target_column, test_data_frame)
target_values = np.concatenate((train_target_values, test_target_values))

# Tokenize data frame
sample_corpus = get_corpus(sample_data_frame)
train_corpus = tokenize_sentence(train_data_frame)
test_corpus = tokenize_sentence(test_data_frame)
corpus = train_corpus + test_corpus

# Vectorized - BOW
bag_of_words = BoW(corpus)
vectors_bag_of_words = bag_of_words.vectorize_text()

# Vectorized - Word2Vec
news_title_array = train_data_frame["text"].values
news_title_array = test_data_frame["text"].values
sss = np.concatenate((news_title_array, news_title_array))
print('news_title_array', sss)
tokenized_sentences = [nltk.word_tokenize(title) for title in news_title_array]
word_2_vec = Word2VecModel(tokenized_sentences)
vectors_word_2_vec = word_2_vec.vectorize_text()

# Split Train-Test Data
X_train, X_test, y_train, y_test = train_test_split(vectors_bag_of_words, target_values, test_size=0.3, random_state=32)

# Logistic Regression
logistic_regression_model = LogisticRegressionModel(X_train, X_test, y_train, y_test)
logistic_regression_y_predict = logistic_regression_model.results()

ComposeMetrics(y_test, logistic_regression_y_predict, 'Logistic Regression Model', [0, 1, 2])

# X_train, X_test, y_train, y_test = train_test_split(vectors_word_2_vec, target_values, test_size=0.3, random_state=109)
#
# logistic_regression_model2 = LogisticRegressionModel(X_train, X_test, y_train, y_test)
# logistic_regression_model_results2 = logistic_regression_model2.results()
# print(logistic_regression_model_results2)
