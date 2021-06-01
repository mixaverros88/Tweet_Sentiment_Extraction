import re
import string
import os
from datetime import datetime


class DataCleaning:
    # class attributes
    initial_data_frame = ''

    def __init__(self, data_frame, column_name):
        self.column_name = column_name
        self.data_frame = data_frame
        self.initial_data_frame = data_frame.copy()

    def data_cleaning(self):
        self.drop_row_if_has_null_column()
        self.remove_column_from_data_frame()
        self.sanitize_data_frame()
        self.compare_dataframes()
        return self.data_frame

    def drop_row_if_has_null_column(self):
        """Drop All Rows with any Null/NaN/NaT Values"""
        self.data_frame.dropna(inplace=True)

    def remove_column_from_data_frame(self):
        """Remove a column from give data frame """
        self.data_frame.drop(self.column_name, axis=1, inplace=True)

    def sanitize_data_frame(self):
        for index, row in self.data_frame.iterrows():
            text = self.trim_text(
                self.covert_text_to_lower_case(
                    self.clean_html(
                        self.remove_punctuations_from_a_string(row['text'])
                    )
                )
            )
            # print(row['text'] + ' --- ' + text)
            self.data_frame.loc[index, 'text'] = text
        # print(self.data_frame)

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

    def compare_dataframes(self):
        print('initial ')
        for index, row in self.initial_data_frame.head(5).iterrows():
            print(row['text'])
        print('cleaned ')
        for index, row in self.data_frame.head(5).iterrows():
            print(row['text'])
