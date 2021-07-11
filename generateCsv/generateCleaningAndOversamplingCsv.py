import utils.dataset as data
from apiService.utils.clean_dataset.DataCleaning import DataCleaning
from utils.VisualizeDataset import VisualizeDataset
from utils.LabelEncoderTransform import LabelEncoderTransform
from utils.BalanceDataset import BalanceDataset

train_dataset = 'Train_Dataset'
test_dataset = 'Test_Dataset'
target_name = 'sentiment'
text_id_column = 'textID'

# Retrieve Data Frames
train_data_frame = data.read_train_data_set()
test_data_frame = data.read_test_data_set()

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
train_data_frame_oversampling = train_balance_dataset.over_sampling_majority_class()

# Visualize Balance Dataset
VisualizeDataset(train_data_frame_oversampling, train_dataset + ' Over Sampling ', target_name,
                 'count_plot_target_class_train_df_balance_over_sampling')

# Cleaning Dataset
train_cleaning_dataset = DataCleaning(train_data_frame_oversampling, text_id_column, train_dataset + 'Over_Sampling')
cleaned_train_data_frame = train_cleaning_dataset.data_pre_processing()

test_cleaning_dataset = DataCleaning(test_data_frame, text_id_column, test_dataset)
cleaned_test_data_frame = test_cleaning_dataset.data_pre_processing()
