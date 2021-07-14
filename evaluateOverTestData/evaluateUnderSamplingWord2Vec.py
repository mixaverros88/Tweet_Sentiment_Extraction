import utils.dataset as data
from utils.serializedModels import word2vec_logistic_regression_under_sampling, word2vec_svm_under_sampling, \
    word2vec_multi_layer_perceptron_classifier_under_sampling, \
    word2vec_decision_tree_under_sampling, word2vec_k_neighbors_under_sampling, word2vec_under_sampling
from sklearn.metrics import classification_report
from definitions import ROOT_DIR
from utils.functions import convert_data_frame_sentence_to_vector_array
import configparser

config = configparser.RawConfigParser()
config.read(ROOT_DIR + '/ConfigFile.properties')
target_column = config.get('STR', 'target.column')
data_set = config.get('STR', 'data.under.sampling')
word_embedding = config.get('STR', 'word.embedding.bow')
test_size = float(config.get('PROJECT', 'test.size'))
random_state = int(config.get('PROJECT', 'random.state'))
remove_words_by_occur_size = int(config.get('PROJECT', 'remove.words.occur.size'))
remove_most_common_word_size = int(config.get('PROJECT', 'remove.most.common.word'))

# Retrieve Data Frames
test_data_set = data.read_cleaned_test_data_set()

# Remove Null rows
test_data_set.dropna(inplace=True)

word2vec_model = word2vec_under_sampling()

X = convert_data_frame_sentence_to_vector_array(word2vec_model, test_data_set)
y = test_data_set['sentiment']

# Logistic Regression
lg = word2vec_logistic_regression_under_sampling()
print("Logistic Regression")
print(classification_report(y, lg.predict(X)))

# Support Vector Machine
svm = word2vec_svm_under_sampling()
print("Support Vector Machine")
print(classification_report(y, svm.predict(X)))

# TODO: Gaussian Naive Bayes

# MLP Classifier
mpl = word2vec_multi_layer_perceptron_classifier_under_sampling()
print("MLP Classifier")
print(classification_report(y, mpl.predict(X)))

# Decision Tree
dt = word2vec_decision_tree_under_sampling()
print("Decision Tree")
print(classification_report(y, dt.predict(X)))

# KNeighbors
kn = word2vec_k_neighbors_under_sampling()
print("KNeighbors")
print(classification_report(y, kn.predict(X)))
