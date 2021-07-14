import utils.dataset as data
from models.text_vectorization.Tfidf import Tfidf
from sklearn.metrics import classification_report
from utils.functions import tokenize_sentence, count_word_occurrences, remove_words_from_corpus, \
    count_the_most_common_words_in_data_set_convert, count_the_most_common_words_in_data_set
from utils.serializedModels import tfidf_logistic_regression_under_sampling, tfidf_svm_under_sampling, \
    tfidf_decision_tree_under_sampling, tfidf_k_neighbors_under_sampling, tfidf_nb_under_sampling, \
    tfidf_multi_layer_perceptron_classifier_under_sampling
from definitions import ROOT_DIR
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

# List of words that occurs 3 or less times
list_of_words_tha_occurs_3_or_less_times = count_word_occurrences(test_data_set,
                                                                  remove_words_by_occur_size)

# List of top 15 most common word
most_common_words = count_the_most_common_words_in_data_set(test_data_set, 'text',
                                                            remove_most_common_word_size)
most_common_words = count_the_most_common_words_in_data_set_convert(most_common_words)

# Tokenize data frame
corpus = tokenize_sentence(test_data_set)

# Remove from corpus the given list of words
corpus = remove_words_from_corpus(corpus, list_of_words_tha_occurs_3_or_less_times + most_common_words)

# Vectorized - TF-IDF
tfidf_of_words_under_sampling = Tfidf(corpus)
vectors_bag_of_words_under_sampling = tfidf_of_words_under_sampling.text_vectorization_test_data_set_under_sampling()

X = vectors_bag_of_words_under_sampling
y = test_data_set['sentiment']

# Logistic Regression
lg = tfidf_logistic_regression_under_sampling()
print("Logistic Regression")
print(classification_report(y, lg.predict(X)))

# Support Vector Machine
svm = tfidf_svm_under_sampling()
print("Support Vector Machine")
print(classification_report(y, svm.predict(X)))

# Gaussian Naive Bayes
gn = tfidf_nb_under_sampling()
print("Gaussian Naive Bayes")
print(classification_report(y, gn.predict(X)))

# MLP Classifier
mpl = tfidf_multi_layer_perceptron_classifier_under_sampling()
print("MLP Classifier")
print(classification_report(y, mpl.predict(X)))

# Decision Tree
dt = tfidf_decision_tree_under_sampling()
print("Decision Tree")
print(classification_report(y, dt.predict(X)))

# KNeighbors
kn = tfidf_k_neighbors_under_sampling()
print("KNeighbors")
print(classification_report(y, kn.predict(X)))
