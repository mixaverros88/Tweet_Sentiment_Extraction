import re
import unicodedata

from helper_functions.clean_dataset.contractions import CONTRACTION_MAP


class DataCleaning:
    # class attributes
    initial_data_frame = ''
    text = ''

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
            self.text = row['text']
            self.text = self.remove_punctuations_special_characters()
            self.text = self.remove_urls()
            self.text = self.remove_html_tags()
            self.text = self.remove_emojis()
            self.text = self.remove_accented_chars()
            self.text = self.expand_contractions()
            self.text = self.trim_text()
            self.text = self.covert_text_to_lower_case()

            # print(row['text'] + ' --- ' + text)
            self.data_frame.loc[index, 'text'] = self.text
        # print(self.data_frame)

    def remove_punctuations_special_characters(self):
        punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
        text_without_punctuations = ''
        for char in self.text:
            if char not in punctuations:
                text_without_punctuations = text_without_punctuations + char
        return text_without_punctuations

    def remove_html_tags(self):
        cleaner = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
        return re.sub(cleaner, '', self.text)

    def remove_urls(self):
        cleaner = re.compile('http\S+')
        return re.sub(cleaner, '', self.text)

    def covert_text_to_lower_case(self):
        return self.text.lower()

    def trim_text(self):
        return self.text.strip()

    def remove_accented_chars(self):
        return unicodedata.normalize('NFKD', self.text).encode('ascii', 'ignore').decode('utf-8', 'ignore')

    def compare_dataframes(self):
        print('initial ')
        for index, row in self.initial_data_frame.head(20).iterrows():
            print(row['text'])
        print('cleaned ')
        for index, row in self.data_frame.head(20).iterrows():
            print(row['text'])

    def remove_emojis(self):
        regrex_pattern = \
            re.compile(pattern="["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               "]+", flags=re.UNICODE)
        return regrex_pattern.sub(r'', self.text)

    def expand_contractions(self):
        contraction_mapping = CONTRACTION_MAP
        contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())),
                                          flags=re.IGNORECASE | re.DOTALL)

        def expand_match(contraction):
            match = contraction.group(0)
            first_char = match[0]
            expanded_contraction = contraction_mapping.get(match) \
                if contraction_mapping.get(match) \
                else contraction_mapping.get(match.lower())
            expanded_contraction = first_char + expanded_contraction[1:]
            return expanded_contraction

        expanded_text = contractions_pattern.sub(expand_match, self.text)
        expanded_text = re.sub("'", "", expanded_text)
        return expanded_text
