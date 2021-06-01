from helper_functions.read_dataset import functions as read_dataset
from helper_functions.clean_dataset.DataCleaning import DataCleaning
from helper_functions.visualize.VisualizeDataset import VisualizeDataset
from helper_functions.tokenize.RemoveStopWords import RemoveStopWords

# Retrieve data frames
train_data_frame = read_dataset.read_train_data_set()
test_data_frame = read_dataset.read_test_data_set()

# Visualize Dataset
VisualizeDataset(train_data_frame, 'sentiment', 'count_plot_target_class_train_df')
VisualizeDataset(test_data_frame, 'sentiment', 'count_plot_target_class_test_df')

# Cleaning Dataset
train_cleaning_dataset = DataCleaning(train_data_frame, 'textID')
cleaned_train_data_frame = train_cleaning_dataset.data_cleaning()

test_cleaning_dataset = DataCleaning(test_data_frame, 'textID')
cleaned_test_data_frame = test_cleaning_dataset.data_cleaning()

# Remove Stop Words
remove_stop_words_on_train_dataset = RemoveStopWords(cleaned_train_data_frame)
train_corpus = remove_stop_words_on_train_dataset.remove_stop_words()

remove_stop_words_on_test_dataset = RemoveStopWords(cleaned_test_data_frame)
test_corpus = remove_stop_words_on_test_dataset.remove_stop_words()

# Vectorized text and target class


# print(train_corpus)
# print(cleaned_train_data_frame)
# print(len(train_corpus))
