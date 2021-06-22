from helper_functions.retrieve import dataset as read_dataset
from helper_functions.tokenizer.functions import get_column_values_as_np_array, tokenize_sentence
from helper_functions.text_vectorization.BoW import BoW
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
train_data_frame_under_sampling = read_dataset.read_cleaned_train_data_set_under_sampling()
test_data_frame = read_dataset.read_cleaned_test_data_set()

# TODO: check why nulls rows
# Remove Null rows
train_data_frame_under_sampling.dropna(inplace=True)
test_data_frame.dropna(inplace=True)

# Get Target Values as Numpy Array
target_values = get_column_values_as_np_array(target_column, train_data_frame_under_sampling)

# Tokenize data frame
corpus = tokenize_sentence(train_data_frame_under_sampling)

# Vectorized - BOW
bag_of_words_under_sampling = BoW(corpus, config.get('MODELS', 'under_sampling.BOW.bow'))
vectors_bag_of_words_under_sampling = bag_of_words_under_sampling.vectorize_text()

# Split Train-Test Data
X_train, X_test, y_train, y_test = train_test_split(vectors_bag_of_words_under_sampling, target_values, test_size=0.33,
                                                    random_state=32)

# Logistic Regression
logistic_regression_model = LogisticRegressionModel(X_train, X_test, y_train, y_test,
                                                    config.get('MODELS', 'under_sampling.BOW.lg'))
logistic_regression_y_predict = logistic_regression_model.results()

ComposeMetrics(y_test, logistic_regression_y_predict, 'Logistic Regression Model', [0, 1, 2])

# Support Vector Machine
svm_model = SvmModel(X_train, X_test, y_train, y_test, config.get('MODELS', 'under_sampling.BOW.svm'))
svm_y_predict = svm_model.results()

ComposeMetrics(y_test, svm_y_predict, 'SVM Model', [0, 1, 2])

# Gaussian Naive Bayes
nb_model = GaussianNBModel(X_train, X_test, y_train, y_test, config.get('MODELS', 'under_sampling.BOW.gaussian'))
nb_y_predict = nb_model.results()

ComposeMetrics(y_test, nb_y_predict, 'NB Model', [0, 1, 2])

# MLP Classifier
neural_network = MLPClassifierModel(X_train, X_test, y_train, y_test, config.get('MODELS', 'under_sampling.BOW.mlp'))
neural_network_predict = neural_network.results()

ComposeMetrics(y_test, neural_network_predict, 'MLPClassifier Model', [0, 1, 2])

decision_tree = DecisionTreeModel(X_train, X_test, y_train, y_test, config.get('MODELS', 'under_sampling.BOW.dt'))
decision_tree_predict = decision_tree.results()

ComposeMetrics(y_test, decision_tree_predict, 'Decision Tree Model', [0, 1, 2])

# K Neighbors

kneighbors_model = KNeighborsModel(X_train, X_test, y_train, y_test,
                                   config.get('MODELS', 'under_sampling.BOW.k_neighbors'))
kneighbors_model_predict = kneighbors_model.results()

# ComposeMetrics(y_test, kneighbors_model, 'KNeighbors Classifier', [0, 1, 2])
