import unicodedata
from helper_functions.clean_dataset.contractionList import contractions_list
from helper_functions.clean_dataset.slanglist import slang_list
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer
from autocorrect import Speller
import pandas as pd
import re
import os
from pathlib import Path
from num2words import num2words
from nltk.tokenize.treebank import TreebankWordDetokenizer
import nltk
import en_core_web_lg
from wordsegment import load, segment

nlp = en_core_web_lg.load()

path = Path()
nltk.download('maxent_ne_chunker')
nltk.download('words')


class DataCleaning:
    # class attributes
    initial_data_frame = ''
    text = ''

    def __init__(self, data_frame, column_name=None, dataframe_name=None):
        self.column_name = column_name
        self.dataframe_name = dataframe_name
        self.data_frame = data_frame
        self.initial_data_frame = data_frame.copy()

    def data_pre_processing(self):
        self.drop_row_if_has_null_column()
        if self.column_name is not None:  # If the request is from API
            self.remove_column_from_data_frame()
        self.sanitize_data_frame()
        if self.dataframe_name is not None:  # If the request is from API
            self.create_new_csv()
            self.compare_dataframes()
        return self.data_frame

    def drop_row_if_has_null_column(self):
        """Drop All Rows with any Null/NaN/NaT Values"""
        self.data_frame.dropna(inplace=True)

    def remove_column_from_data_frame(self):
        """Remove a column from give data frame """
        self.data_frame.drop(self.column_name, axis=1, inplace=True)

    def sanitize_data_frame(self):
        """For every row of the data frame proceed with the following steps"""
        for index, row in self.data_frame.iterrows():
            self.text = row['text']
            print('#### START: ' + str(index) + ' ####')
            print('1. Initial Text: ' + self.text)
            self.text = self.name_entity_recognition()
            print('2. name_entity_recognition: ' + self.text)
            self.text = self.expand_contractions()
            print('3. expand_contractions: ' + self.text)
            self.text = self.remove_am_pm_dates()
            print('4. remove_am_pm_dates: ' + self.text)
            self.text = self.remove_emojis()
            print('5. remove_emojis: ' + self.text)
            self.text = self.covert_text_to_lower_case()
            print('6. covert_text_to_lower_case: ' + self.text)
            self.text = self.replace_slang_word()
            print('7. replace_slang_word: ' + self.text)
            self.text = self.remove_urls()
            print('8. remove_urls: ' + self.text)
            self.text = self.remove_html_tags()
            print('9. remove_html_tags: ' + self.text)
            self.text = self.remove_hash_tags()
            print('10. remove_hash_tags: ' + self.text)
            self.text = self.convert_accented_characters_to_ASCII_characters()
            print('11. remove_accented_chars: ' + self.text)
            self.text = self.remove_punctuations_special_characters()
            print('12. remove_punctuations_special_characters: ' + self.text)
            self.text = self.remove_consequently_char()
            print('13. remove_consequently_char: ' + self.text)
            self.text = self.replace_slang_word()
            print('14. replace_slang_word: ' + self.text)
            self.text = self.trim_text()
            print('15. trim_text: ' + self.text)
            self.text = self.remove_double_spaces()
            print('16. remove_double_spaces: ' + self.text)
            # self.text = self.word_segment()
            # print('27. word_segment: ' + self.text)
            self.text = self.auto_spelling()
            print('18. auto_spelling: ' + self.text)
            self.text = self.lemmatization()
            print('19. lemmatization: ' + self.text)
            self.text = self.convert_num_to_word()
            print('20. convert_num_to_word: ' + self.text)
            self.text = self.convert_to_nominal()
            print('21. convert_to_nominal: ' + self.text)
            print('#### STOP: ' + str(index) + ' #### \n')
            self.data_frame.loc[index, 'text'] = self.text

    def remove_punctuations_special_characters(self):
        punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
        text_without_punctuations = ''
        for char in self.text:
            if char not in punctuations:
                text_without_punctuations = text_without_punctuations + char
            else:
                text_without_punctuations = text_without_punctuations + " "
        return text_without_punctuations

    def remove_am_pm_dates(self):
        cleaner = re.compile('(?:\d{2}|\d{1})(?:AM|PM|am|pm)')
        return re.sub(cleaner, '', self.text)

    def remove_html_tags(self):
        """ Remove HTML tags e.g. <div> """
        cleaner = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
        return re.sub(cleaner, '', self.text)

    def remove_urls(self):
        """ Remove HTML tags e.g. https://scikit-learn.org/stable/modules/neural_networks_supervised.html"""
        cleaner = re.compile('http\S+')
        return re.sub(cleaner, '', self.text)

    def remove_hash_tags(self):
        return ' '.join(re.sub('(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)', ' ', self.text).split())

    def covert_text_to_lower_case(self):
        return self.text.lower()

    def trim_text(self):
        return self.text.strip()

    def remove_double_spaces(self):
        return re.sub('\s+', ' ', self.text)

    def convert_accented_characters_to_ASCII_characters(self):
        return unicodedata.normalize('NFKD', self.text).encode('ascii', 'ignore').decode('utf-8', 'ignore')

    def create_new_csv(self):
        """ Summarize the raw tweet with the cleaned tweet in one csv in order to analyze the results"""
        merge_dataframes = pd.concat([self.data_frame['text'], self.data_frame['sentiment']], axis=1,
                                     keys=['text', 'sentiment'])
        merge_dataframes.to_csv(os.path.abspath(
            path.parent.absolute().parent) + '\\datasets\\cleaned\\' + self.dataframe_name + "_dataframe_cleaned.csv",
                                sep=',', index=False, header=True)

    def compare_dataframes(self):
        merge_dataframes = pd.concat([self.initial_data_frame['text'], self.data_frame['text']], axis=1,
                                     keys=['Initial Text', 'Cleaned Text'])
        merge_dataframes.to_csv(os.path.abspath(
            path.parent.absolute().parent) + '\\presentation\\results\\' + self.dataframe_name + "_dataframe_cleaned_initial.csv",
                                sep=',', encoding='utf-8', index=False)

    def remove_emojis(self):
        """ remove emoji e.g. 👋 """
        regrex_pattern = \
            re.compile("["
                       u"\U0001F600-\U0001F64F"  # emoticons
                       u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                       u"\U0001F680-\U0001F6FF"  # transport & map symbols
                       u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                       u"\U0001F1F2-\U0001F1F4"  # Macau flag
                       u"\U0001F1E6-\U0001F1FF"  # flags
                       u"\U0001F600-\U0001F64F"
                       u"\U00002702-\U000027B0"
                       u"\U000024C2-\U0001F251"
                       u"\U0001f926-\U0001f937"
                       u"\U0001F1F2"
                       u"\U0001F1F4"
                       u"\U0001F620"
                       u"\u200d"
                       u"\u2640-\u2642"
                       "]+", flags=re.UNICODE)
        return regrex_pattern.sub(r'', self.text)

    def expand_contractions(self):
        for word in self.text.split():
            if word.lower() in contractions_list:
                self.text = self.text.replace(word, contractions_list[word.lower()])
        return self.text

    def remove_consequently_char(self):
        return re.sub(r'(.)\1+', r'\1\1', self.text)

    def replace_slang_word(self):
        for word in self.text.split():
            if word.lower() in slang_list:
                self.text = self.text.replace(word, slang_list[word.lower()])
        return self.text

    def auto_spelling(self):
        spell = Speller(lang='en')
        spells = [spell(w) for w in (word_tokenize(self.text))]
        return " ".join(spells)

    def lemmatization(self):
        wordnet_lemmatizer = WordNetLemmatizer()
        lemmatized_word = [wordnet_lemmatizer.lemmatize(word) for sent in sent_tokenize(self.text) for word in
                           word_tokenize(sent)]
        return " ".join(lemmatized_word)

    def convert_num_to_word(self):
        """e.g 1 to one"""
        tokens = nltk.word_tokenize(self.text)
        tokens = nltk.pos_tag(tokens)
        text = []
        print(tokens)
        for word in tokens:
            if word[1] == 'CD':
                try:
                    text.append(num2words(word[0]))
                except:
                    text.append(word[0])
            else:
                text.append(word[0])
        return TreebankWordDetokenizer().detokenize(text)

    def convert_to_nominal(self):
        """e.g 1st to first"""
        numbers = re.findall('(\d+)[st|nd|rd|th]', self.text)

        newText = self.text
        for n in numbers:
            ordinalAsString = num2words(n, ordinal=True)
            newText = re.sub(r"\d+[st|nd|rd|th]", ordinalAsString[:-1], self.text, 1)
            print(newText)
        print(self.text)
        print(newText)
        return newText

    def word_segment(self):
        """Segment Words e.g loveit to love it"""
        load()
        return ' '.join(segment(self.text))

    # https://towardsdatascience.com/named-entity-recognition-with-nltk-and-spacy-8c4a7d88e7da
    # Named Entity Recognition (NER)
    def name_entity_recognition(self):
        doc = nlp(self.text)
        print([(X.text, X.label_) for X in doc.ents])
        for idx, X in enumerate(doc.ents):
            # REMOVE organizations
            if X.label_ == 'ORG' or X.label_ == 'PERSON' or X.label_ == 'GPE' or X.label_ == 'TIME':
                self.text = self.text.replace(X.text, '')
        print(self.text)
        return self.text
