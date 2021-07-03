from helper.retrieve import dataset as read_dataset
from helper.helper_functions.functions import tokenize_sentence
from models.text_vectorization.BoW import BoW
from models.machine_learning import LogisticRegressionModel
from models.machine_learning import SvmModel
from models.machine_learning import GaussianNBModel
from models.machine_learning import DecisionTreeModel
from models.neural import MLPClassifierModel

from sklearn.metrics import classification_report
import configparser

config = configparser.RawConfigParser()
config.read('../ConfigFile.properties')
target_column = config.get('STR', 'target.column')
data_set = config.get('STR', 'data.over.sampling')
word_embedding = config.get('STR', 'word.embedding.bow')
test_size = float(config.get('PROJECT', 'test.size'))
random_state = int(config.get('PROJECT', 'random.state'))
remove_words_by_occur_size = int(config.get('PROJECT', 'remove.words.occur.size'))

# Retrieve Data Frames
test_data_set = read_dataset.read_cleaned_test_data_set()

# Remove Null rows
test_data_set.dropna(inplace=True)

# Tokenize data frame
corpus = tokenize_sentence(test_data_set)

# Vectorized - BOW
bag_of_words_over_sampling = BoW(corpus)
vectors_bag_of_words_over_sampling = bag_of_words_over_sampling.text_vectorization_test_data_set()

X = vectors_bag_of_words_over_sampling
y = test_data_set['sentiment']

# Logistic Regression
logistic_regression_model = LogisticRegressionModel.run_on_test_data_set(X, y)

print(classification_report(y, logistic_regression_model))

# Support Vector Machine
svm = SvmModel.run_on_test_data_set(X, y)
print(classification_report(y, svm))

# Gaussian Naive Bayes
gn = GaussianNBModel.run_on_test_data_set(X, y)
print(classification_report(y, gn))

# MLP Classifier
mpl = MLPClassifierModel.run_on_test_data_set(X, y)
print(classification_report(y, mpl))

# Decision Tree
mpl = DecisionTreeModel.run_on_test_data_set(X, y)
print(classification_report(y, mpl))

# TODO: KNeighborsModel