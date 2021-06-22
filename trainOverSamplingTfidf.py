from helper_functions.retrieve import dataset as read_dataset
from helper_functions.tokenizer.functions import get_column_values_as_np_array, tokenize_sentence, \
    count_word_occurrences, remove_words_from_corpus
from helper_functions.text_vectorization.Tfidf import Tfidf
from helper_functions.metrics.ComposeMetrics import ComposeMetrics
from models.machine_learning.LogisticRegressionModel import LogisticRegressionModel
from models.machine_learning.SvmModel import SvmModel
from models.machine_learning.GaussianNBModel import GaussianNBModel
from models.neural.MLPClassifierModel import MLPClassifierModel
from models.machine_learning.KNeighborsModel import KNeighborsModel
from models.machine_learning.DecisionTreeModel import DecisionTreeModel
from sklearn.model_selection import train_test_split
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

word_list = count_word_occurrences(train_data_frame_over_sampling, 1)

# Tokenize data frame
corpus = tokenize_sentence(train_data_frame_over_sampling)

corpus = remove_words_from_corpus(corpus, word_list)

# Vectorized - BOW
tfidf_of_words_over_sampling = Tfidf(corpus, config.get('MODELS', 'oversampling.Tfidf.bow'))
vectors_bag_of_words_over_sampling = tfidf_of_words_over_sampling.vectorize_text()

# Split Train-Test Data
X_train, X_test, y_train, y_test = train_test_split(vectors_bag_of_words_over_sampling,
                                                    target_values, test_size=0.3, random_state=32)

# Logistic Regression
logistic_regression_model = LogisticRegressionModel(X_train, X_test, y_train, y_test,
                                                    config.get('MODELS', 'oversampling.Tfidf.lg'))
logistic_regression_y_predict = logistic_regression_model.results()

ComposeMetrics(y_test, logistic_regression_y_predict, 'Logistic Regression Model', [0, 1, 2])

# Support Vector Machine
svm_model = SvmModel(X_train, X_test, y_train, y_test, config.get('MODELS', 'oversampling.Tfidf.svm'))
svm_y_predict = svm_model.results()

ComposeMetrics(y_test, svm_y_predict, 'SVM Model', [0, 1, 2])

# Gaussian Naive Bayes
nb_model = GaussianNBModel(X_train, X_test, y_train, y_test, config.get('MODELS', 'oversampling.Tfidf.gaussian'))
nb_y_predict = nb_model.results()

ComposeMetrics(y_test, nb_y_predict, 'NB Model', [0, 1, 2])

# MLP Classifier
neural_network = MLPClassifierModel(X_train, X_test, y_train, y_test, config.get('MODELS', 'oversampling.Tfidf.mlp'))
neural_network_predict = neural_network.results()

ComposeMetrics(y_test, neural_network_predict, 'MLPClassifier Model', [0, 1, 2])

decision_tree = DecisionTreeModel(X_train, X_test, y_train, y_test, config.get('MODELS', 'oversampling.Tfidf.dt'))
decision_tree_predict = decision_tree.results()

# ComposeMetrics(y_test, decision_tree_predict, 'Decision Tree Model', [0, 1, 2])

# K Neighbors
k_neighbors_model = KNeighborsModel(X_train, X_test, y_train, y_test, config.get('MODELS', 'oversampling.Tfidf.k_neighbors'))
k_neighbors_model_predict = k_neighbors_model.results()

# ComposeMetrics(y_test, k_neighbors_model_predict, 'KNeighbors Classifier', [0, 1, 2])
