import matplotlib.pyplot as plt
import seaborn as sns
import os
from helper.helper_functions.functions import count_words_per_sentence, count_the_most_common_words_in_data_set, \
    sanitize_model_name
from pathlib import Path
import numpy as np
path = Path()


class VisualizeDataset:

    def __init__(self, data_frame, dataframe_name, target_column, generated_image_name):
        self.data_frame = data_frame
        self.dataframe_name = dataframe_name
        self.target_column = target_column
        self.generated_image_name = generated_image_name
        self.folder = 'visualize_data_set'
        self.count_plot_target_class()
        self.check_for_null_values()
        self.data_frame_info()
        self.count_plot_word_length()
        self.count_plot_most_common_words_in_data_set()

    def count_plot_target_class(self):
        """Generate an image regarding the distribution of given target column of a given dataframe"""
        print(self.dataframe_name)
        print(self.data_frame.groupby([self.target_column]).size())  # print the sum of every class

        sns.countplot(data=self.data_frame, x=self.data_frame[self.target_column])
        plt.title(self.dataframe_name + ': Display the distribution of ' + self.target_column + ' class')
        plt.xlabel('Target Name: ' + self.target_column)
        plt.ylabel('Count')
        self.save_plot_as_image()
        plt.show()

    def save_plot_as_image(self):
        """Save the plot as image"""
        plt.savefig(os.path.abspath(
            path.parent.absolute().parent) + '\\presentation\\images\\' + self.folder + '\\' + self.generated_image_name + '.png',
                    bbox_inches='tight')

    def data_frame_info(self):
        """Display Data Frame information"""
        print(self.dataframe_name)
        print(self.data_frame.info())

    def check_for_null_values(self):
        """Count for every column the null occurrences"""
        print(self.dataframe_name)
        print(self.data_frame.isna().sum())

    # https://towardsdatascience.com/the-real-world-as-seen-on-twitter-sentiment-analysis-part-one-5ac2d06b63fb
    def count_plot_word_length(self):
        self.data_frame['word count'] = \
            self.data_frame.apply(lambda row: count_words_per_sentence(row['text']), axis=1)
        # plot word count distribution for both positive and negative sentiments
        if self.data_frame['sentiment'].dtype == 'int32':
            count_positive = self.data_frame['word count'][self.data_frame['sentiment'] == 2]
            count_negative = self.data_frame['word count'][self.data_frame['sentiment'] == 0]
            count_neutral = self.data_frame['word count'][self.data_frame['sentiment'] == 1]
        else:
            count_positive = self.data_frame['word count'][self.data_frame['sentiment'] == 'positive']
            count_negative = self.data_frame['word count'][self.data_frame['sentiment'] == 'negative']
            count_neutral = self.data_frame['word count'][self.data_frame['sentiment'] == 'neutral']

        plt.figure(figsize=(12, 6))
        plt.xlim(0, 45)
        plt.xlabel('word count')
        plt.ylabel('frequency')
        plt.title(self.dataframe_name + ': Display the word length')
        plt.hist(
            [count_positive, count_negative, count_neutral],
            color=['b', 'r', 'g'],
            alpha=0.5,
            label=['positive', 'negative', 'neutral'])
        plt.legend(loc='upper right')
        self.generated_image_name = sanitize_model_name(self.dataframe_name) + 'count_plot_word_length'
        self.save_plot_as_image()
        plt.show()
        del self.data_frame['word count']

    # https://towardsdatascience.com/the-real-world-as-seen-on-twitter-sentiment-analysis-part-one-5ac2d06b63fb
    def count_plot_most_common_words_in_data_set(self):
        list_of_most_common_words = count_the_most_common_words_in_data_set(self.data_frame, 'text', 20)
        print(list_of_most_common_words)
        word = []
        frequency = []
        for i in range(len(list_of_most_common_words)):
            word.append(list_of_most_common_words[i][0])
            frequency.append(list_of_most_common_words[i][1])

        indices = np.arange(len(list_of_most_common_words))
        plt.bar(indices, frequency, color='r')
        plt.xticks(indices, word)
        plt.tight_layout()
        plt.title(self.dataframe_name + ': Display the most common words in data set')
        self.generated_image_name = sanitize_model_name(self.dataframe_name) + 'count_plot_most_common_words'
        self.save_plot_as_image()
        plt.show()
