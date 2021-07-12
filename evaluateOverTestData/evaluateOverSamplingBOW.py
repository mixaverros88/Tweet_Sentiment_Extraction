import utils.dataset as data
from models.text_vectorization.BoW import BoW
from models.machine_learning import DecisionTreeModel, GaussianNBModel, SvmModel, LogisticRegressionModel
from models.neural import MLPClassifierModel
from sklearn.metrics import classification_report
from utils.functions import tokenize_sentence, count_word_occurrences, remove_words_from_corpus, \
    count_the_most_common_words_in_data_set_convert, count_the_most_common_words_in_data_set
from definitions import ROOT_DIR
import configparser

config = configparser.RawConfigParser()
config.read(ROOT_DIR + '/ConfigFile.properties')
target_column = config.get('STR', 'target.column')
data_set = config.get('STR', 'data.over.sampling')
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

# Vectorized - BOW
bag_of_words_over_sampling = BoW(corpus)
vectors_bag_of_words_over_sampling = bag_of_words_over_sampling.text_vectorization_test_data_set()

X = vectors_bag_of_words_over_sampling
y = test_data_set['sentiment']

# Logistic Regression
logistic_regression_model = LogisticRegressionModel.run_on_test_data_set(X, y)
print("Logistic Regression")
print(classification_report(y, logistic_regression_model))

# Support Vector Machine
svm = SvmModel.run_on_test_data_set(X, y)
print("Support Vector Machine")
print(classification_report(y, svm))

# Gaussian Naive Bayes
gn = GaussianNBModel.run_on_test_data_set(X, y)
print("Gaussian Naive Bayes")
print(classification_report(y, gn))

# MLP Classifier
mpl = MLPClassifierModel.run_on_test_data_set(X, y)
print("MLP Classifier")
print(classification_report(y, mpl))

# Decision Tree
mpl = DecisionTreeModel.run_on_test_data_set(X, y)
print("Decision Tree")
print(classification_report(y, mpl))

# TODO: KNeighborsModel
