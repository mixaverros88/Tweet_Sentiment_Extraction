import re
import string
import os
from datetime import datetime


class DataCleaning:

    def __init__(self, data_frame, column_name):
        self.drop_row_if_has_null_column(data_frame)
        self.remove_column_from_data_frame(data_frame, column_name)
        self.sanitize_data_frame(data_frame)

    def drop_row_if_has_null_column(self, data_frame):
        """Drop All Rows with any Null/NaN/NaT Values"""
        data_frame.dropna(inplace=True)

    def remove_column_from_data_frame(self, data_frame, column_name):
        """Remove a column from give data frame """
        data_frame.drop(column_name, axis=1, inplace=True)

    def sanitize_data_frame(self, data_frame):
        for index, row in data_frame.iterrows():
            text = self.trim_text(
                self.covert_text_to_lower_case(self.clean_html(self.remove_punctuations_from_a_string(row['text']))))
            print(row['text'] + ' --- ' + text)
            data_frame.loc[index, 'text'] = text

    def remove_punctuations_from_a_string(self, text):
        punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
        text_without_punctuations = ''
        for char in text:
            if char not in punctuations:
                text_without_punctuations = text_without_punctuations + char
        return text_without_punctuations

    def clean_html(self, text):
        cleaner = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
        return re.sub(cleaner, '', text)

    def covert_text_to_lower_case(self, text):
        return text.lower()

    def trim_text(self, text):
        return text.strip()

    def generate_results_file(self, text):
        now = datetime.now()  # current date and time
        date_time = now.strftime("%m_%d_%Y_%H_%M_%S")
        print("date and time:", date_time)
        alphabet_char_list = string.ascii_lowercase
        generated_file = open(os.path.abspath(os.curdir) + '\\presentation\\results\\' + date_time + '.txt', "w+",
                              encoding="utf-8")
        generated_file.write(text + '\n')
        generated_file.close()

    # TODO: vectorized target class

    # TODO: Removing stop words
    def remove_stop_word(self):
        return
