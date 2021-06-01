import matplotlib.pyplot as plt
import seaborn as sns


class VisualizeDataset:

    def __init__(self, data_frame, target_column, generated_image_name):
        self.data_frame = data_frame
        self.target_column = target_column
        self.generated_image_name = generated_image_name
        self.count_plot_target_class()
        self.check_for_null_values()
        self.data_frame_info()

    def count_plot_target_class(self):
        """Generate an image regarding the distribution of given target column of a given dataframe"""
        print(self.data_frame.groupby([self.target_column]).size())  # print the sum of every class

        sns.countplot(data=self.data_frame, x=self.data_frame[self.target_column])
        plt.title('Display the distribution of ' + self.target_column + ' class')
        plt.xlabel('Target Name: ' + self.target_column)
        plt.ylabel('Count')
        self.save_plot_as_image()
        plt.show()

    def save_plot_as_image(self):
        """Save the plot as image"""
        plt.savefig('presentation/images/' + self.generated_image_name + '.png', bbox_inches='tight')

    def data_frame_info(self):
        """Display Data Frame information"""
        print(self.data_frame.info())

    def check_for_null_values(self):
        """Count for every column the null occurrences"""
        print(self.data_frame.isna().sum())
