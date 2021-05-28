from helper_functions.read_dataset import functions as read_dataset
from helper_functions.visualize_functions import functions as visualize_functions

train_data_frame = read_dataset.read_train_data_set()
test_data_frame = read_dataset.read_test_data_set()

visualize_functions.count_plot_target_class(train_data_frame, 'sentiment', 'count_plot_target_class_train_df')
visualize_functions.count_plot_target_class(test_data_frame, 'sentiment', 'count_plot_target_class_test_df')

print(train_data_frame)
print(test_data_frame)
