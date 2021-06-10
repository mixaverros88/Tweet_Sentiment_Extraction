from helper_functions.read_dataset import functions as read_dataset
from helper_functions.clean_dataset.DataCleaning import DataCleaning
from helper_functions.visualize.VisualizeDataset import VisualizeDataset
from helper_functions.label_encoder.LabelEncoderTransform import LabelEncoderTransform
from helper_functions.clean_dataset.BalanceDataset import BalanceDataset
from helper_functions.tokenize.TokenizeDataFrame import TokenizeDataFrame
from helper_functions.convert_test.functions import tokenizing_sentences
from helper_functions.text_vectorization.BoW import BoW

train_dataset = 'Train Dataset'
test_dataset = 'Test Dataset'
sample_dataset = 'Sample Dataset'
target_name = 'sentiment'
text_id_column = 'textID'

# Retrieve Data Frames
train_data_frame = read_dataset.read_train_data_set()
sample_data_frame = read_dataset.read_sample_data_set()
test_data_frame = read_dataset.read_test_data_set()

# # Visualize Dataset
VisualizeDataset(train_data_frame, train_dataset, target_name, 'count_plot_target_class_train_df')
VisualizeDataset(test_data_frame, test_dataset, target_name, 'count_plot_target_class_test_df')

# Label Encoder On Target Class
train_label_encoder_transform = LabelEncoderTransform(train_data_frame, target_name)
train_data_frame = train_label_encoder_transform.convert_target_column()

test_label_encoder_transform = LabelEncoderTransform(test_data_frame, target_name)
test_data_frame = test_label_encoder_transform.convert_target_column()

sample_label_encoder_transform = LabelEncoderTransform(sample_data_frame, target_name)
sample_data_frame = sample_label_encoder_transform.convert_target_column()

# Balance Dataset
train_balance_dataset = BalanceDataset(train_data_frame, target_name)
train_data_frame_oversampling = train_balance_dataset.over_sampling_majority_class()
train_data_frame = train_balance_dataset.under_sampling_majority_class()

# Visualize Balance Dataset
VisualizeDataset(train_data_frame, train_dataset + ' Under Sampling', target_name,
                 'count_plot_target_class_train_df_balance_under_sampling')
VisualizeDataset(train_data_frame_oversampling, train_dataset + ' Over Sampling ', target_name,
                 'count_plot_target_class_train_df_balance_over_sampling')
# Cleaning Dataset
sample_cleaning_dataset = DataCleaning(sample_data_frame, text_id_column, sample_dataset)
cleaned_sample_data_frame = sample_cleaning_dataset.data_cleaning()

train_cleaning_dataset = DataCleaning(train_data_frame, text_id_column, train_dataset)
cleaned_train_data_frame = train_cleaning_dataset.data_cleaning()

test_cleaning_dataset = DataCleaning(test_data_frame, text_id_column, test_dataset)
cleaned_test_data_frame = test_cleaning_dataset.data_cleaning()

# Tokenize data frame
tokenize_data_frame_sample_data = TokenizeDataFrame(cleaned_sample_data_frame)
sample_corpus = tokenize_data_frame_sample_data.tokenize_sentence()

tokenize_data_frame_train_data = TokenizeDataFrame(cleaned_train_data_frame)
train_corpus = tokenize_data_frame_train_data.tokenize_sentence()

tokenize_data_frame_test_data = TokenizeDataFrame(cleaned_test_data_frame)
test_corpus = tokenize_data_frame_test_data.tokenize_sentence()

# Convert Text
sample_tokenized_sentences_data_frame = tokenizing_sentences(cleaned_sample_data_frame)
print(sample_tokenized_sentences_data_frame)

# Vectorized Text Column
bag_of_words = BoW(sample_tokenized_sentences_data_frame, sample_corpus)
array = bag_of_words.vectorize_text()
print('array', array)
