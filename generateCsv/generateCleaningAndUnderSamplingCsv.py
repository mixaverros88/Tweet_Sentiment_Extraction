from helper_functions.retrieve import dataset as read_dataset
from helper_functions.clean_dataset.DataCleaning import DataCleaning
from helper_functions.visualize.VisualizeDataset import VisualizeDataset
from helper_functions.label_encoder.LabelEncoderTransform import LabelEncoderTransform
from helper_functions.clean_dataset.BalanceDataset import BalanceDataset

train_dataset = 'Train_Dataset'
test_dataset = 'Test_Dataset'
target_name = 'sentiment'
text_id_column = 'textID'

# Retrieve Data Frames
train_data_frame = read_dataset.read_train_data_set()
test_data_frame = read_dataset.read_test_data_set()

# Visualize Dataset
VisualizeDataset(train_data_frame, train_dataset, target_name, 'count_plot_target_class_train_df')
VisualizeDataset(test_data_frame, test_dataset, target_name, 'count_plot_target_class_test_df')

# Label Encoder On Target Class
train_label_encoder_transform = LabelEncoderTransform(train_data_frame, target_name)
train_data_frame = train_label_encoder_transform.convert_target_column()

test_label_encoder_transform = LabelEncoderTransform(test_data_frame, target_name)
test_data_frame = test_label_encoder_transform.convert_target_column()

# Balance Dataset
train_balance_dataset = BalanceDataset(train_data_frame, target_name)
train_data_frame_under_sampling = train_balance_dataset.under_sampling_majority_class()

# Visualize Balance Dataset
VisualizeDataset(train_data_frame_under_sampling, train_dataset + ' Under Sampling', target_name,
                 'count_plot_target_class_train_df_balance_under_sampling')

# Cleaning Dataset
train_cleaning_dataset = DataCleaning(train_data_frame_under_sampling, text_id_column, train_dataset + 'Under_Sampling')
cleaned_train_data_frame = train_cleaning_dataset.data_cleaning()
