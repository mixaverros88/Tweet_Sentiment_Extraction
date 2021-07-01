from helper.retrieve import dataset as read_dataset
from helper.helper_functions.functions import get_column_values_as_np_array, tokenize_sentence, \
    count_word_occurrences, remove_words_from_corpus
from models.text_vectorization.BoW import BoW
from helper.metrics.ComposeMetrics import ComposeMetrics
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
data_set = config.get('STR', 'data.under.sampling')
word_embedding = config.get('STR', 'word.embedding.bow')
target_column = config.get('STR', 'target.column')
test_size = float(config.get('PROJECT', 'test.size'))
random_state = int(config.get('PROJECT', 'random.state'))
remove_words_by_occur_size = int(config.get('PROJECT', 'remove.words.occur.size'))

# Retrieve Data Frames
train_data_frame_under_sampling = read_dataset.read_cleaned_train_data_set_under_sampling()
test_data_frame = read_dataset.read_cleaned_test_data_set()

# TODO: check why nulls rows
# Remove Null rows
train_data_frame_under_sampling.dropna(inplace=True)
test_data_frame.dropna(inplace=True)

# Get Target Values as Numpy Array
target_values = get_column_values_as_np_array(target_column, train_data_frame_under_sampling)

# List of words tha occurs 3 or less times
word_list = count_word_occurrences(train_data_frame_under_sampling, remove_words_by_occur_size)

# Tokenize data frame
corpus = tokenize_sentence(train_data_frame_under_sampling)

# Remove from corpus the given list of words
corpus = remove_words_from_corpus(corpus, word_list)

# Vectorized - BOW
bag_of_words_under_sampling = BoW(corpus, config.get('MODELS', 'under_sampling.BOW.bow'))
vectors_bag_of_words_under_sampling = bag_of_words_under_sampling.text_vectorization()

# Split Train-Test Data
X_train, X_test, y_train, y_test = train_test_split(vectors_bag_of_words_under_sampling, target_values, test_size=test_size,
                                                    random_state=random_state)

# Logistic Regression
logistic_regression_params = {'C': 1.0, 'penalty': 'l2', 'max_iter': 1000}
logistic_regression_model = LogisticRegressionModel(
    X_train,
    X_test,
    y_train,
    y_test,
    config.get('MODELS', 'under_sampling.BOW.lg'),
    logistic_regression_params)

logistic_regression_y_predict = logistic_regression_model.results()

ComposeMetrics(
    logistic_regression_y_predict.score,
    y_test,
    logistic_regression_y_predict.prediction,
    config.get('MODELNAME', 'model.lg'),
    data_set,
    word_embedding)

# Support Vector Machine
svm_params = {'kernel': 'linear'}
svm_model = SvmModel(
    X_train,
    X_test,
    y_train,
    y_test,
    config.get('MODELS', 'under_sampling.BOW.svm'),
    svm_params)

svm_y_predict = svm_model.results()

ComposeMetrics(
    svm_y_predict.score,
    y_test,
    svm_y_predict.prediction,
    config.get('MODELNAME', 'model.svm'),
    data_set,
    word_embedding)

# Gaussian Naive Bayes
nb_params = {'alpha': 1.5}
nb_model = GaussianNBModel(
    X_train,
    X_test,
    y_train,
    y_test, config.get('MODELS', 'under_sampling.BOW.gaussian'), nb_params)
nb_y_predict = nb_model.results()

ComposeMetrics(
    nb_y_predict.score,
    y_test,
    nb_y_predict.prediction,
    config.get('MODELNAME', 'model.nb'),
    data_set,
    word_embedding)

# MLP Classifier
neural_network_params = {'activation': 'tanh', 'alpha': 0.05, 'hidden_layer_sizes': (5, 5, 5),
                         'learning_rate': 'adaptive', 'max_iter': 1000}
neural_network = MLPClassifierModel(
    X_train,
    X_test,
    y_train,
    y_test,
    config.get('MODELS', 'under_sampling.BOW.mlp'),
    neural_network_params)

neural_network_predict = neural_network.results()

ComposeMetrics(neural_network_predict.score, y_test, neural_network_predict.prediction,
               config.get('MODELNAME', 'model.mlp'), data_set, word_embedding)

# Decision Tree
decision_tree_params = {'max_depth': 5, 'max_leaf_nodes': 18, 'min_samples_split': 3}
decision_tree = DecisionTreeModel(
    X_train,
    X_test,
    y_train,
    y_test,
    config.get('MODELS', 'under_sampling.BOW.dt'),
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
k_neighbors_model = KNeighborsModel(
    X_train,
    X_test,
    y_train,
    y_test,
    config.get('MODELS', 'under_sampling.BOW.k_neighbors'),
    k_neighbors_params)
k_neighbors_model_predict = k_neighbors_model.results()

ComposeMetrics(
    k_neighbors_model_predict.score,
    y_test,
    k_neighbors_model_predict.prediction,
    config.get('MODELNAME', 'model.kn'),
    data_set,
    word_embedding)
