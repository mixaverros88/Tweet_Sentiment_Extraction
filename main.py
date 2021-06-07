from helper_functions.read_dataset import functions as read_dataset
from helper_functions.clean_dataset.DataCleaning import DataCleaning
from helper_functions.visualize.VisualizeDataset import VisualizeDataset
from helper_functions.label_encoder.LabelEncoderTransform import LabelEncoderTransform
from helper_functions.clean_dataset.BalanceDataset import BalanceDataset
from helper_functions.tokenize.RemoveStopWords import RemoveStopWords
from helper_functions.convert_test.functions import tokenizing_sentences
from helper_functions.text_vectorization.BoW import BoW

# Retrieve data frames
train_data_frame = read_dataset.read_train_data_set()
sample_data_frame = read_dataset.read_sample_data_set()
test_data_frame = read_dataset.read_test_data_set()

# Visualize Dataset
VisualizeDataset(train_data_frame, 'Train dataset ', 'sentiment', 'count_plot_target_class_train_df')
VisualizeDataset(test_data_frame, 'Test dataset', 'sentiment', 'count_plot_target_class_test_df')

# Label Encoder On Target Class
train_label_encoder_transform = LabelEncoderTransform(train_data_frame, 'sentiment')
train_data_frame = train_label_encoder_transform.convert_target_column()

test_label_encoder_transform = LabelEncoderTransform(test_data_frame, 'sentiment')
test_data_frame = test_label_encoder_transform.convert_target_column()

# Balance Dataset
train_balance_dataset = BalanceDataset(train_data_frame, 'sentiment')
train_data_frame = train_balance_dataset.convert_to_balance_dataset()

# Visualize Balance Dataset
VisualizeDataset(train_data_frame, 'Train dataset ', 'sentiment', 'count_plot_target_class_train_df_balance')

# Cleaning Dataset
sample_cleaning_dataset = DataCleaning(sample_data_frame, 'textID', 'sample')
cleaned_sample_data_frame = sample_cleaning_dataset.data_cleaning()

train_cleaning_dataset = DataCleaning(train_data_frame, 'textID', 'train')
cleaned_train_data_frame = train_cleaning_dataset.data_cleaning()

test_cleaning_dataset = DataCleaning(test_data_frame, 'textID', 'test')
cleaned_test_data_frame = test_cleaning_dataset.data_cleaning()

# Remove Stop Words
remove_stop_words_on_train_dataset = RemoveStopWords(cleaned_train_data_frame)
train_corpus = remove_stop_words_on_train_dataset.remove_stop_words()

remove_stop_words_on_test_dataset = RemoveStopWords(cleaned_test_data_frame)
test_corpus = remove_stop_words_on_test_dataset.remove_stop_words()
print(cleaned_test_data_frame)

# Convert Text
tokenized_sentences = tokenizing_sentences(cleaned_test_data_frame)
print(tokenized_sentences)

sentences = remove_stop_words_on_train_dataset.tokenize_sentence()
# print(sentences)
# Vectorized text and target class
BoW(sentences)
