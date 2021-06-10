import unicodedata
from helper_functions.clean_dataset.contractionList import contractions_list
from helper_functions.clean_dataset.slanglist import slang_list
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from autocorrect import Speller
import pandas as pd
import re


class DataCleaning:
    # class attributes
    initial_data_frame = ''
    text = ''

    def __init__(self, data_frame, column_name, dataframe_name):
        self.column_name = column_name
        self.dataframe_name = dataframe_name
        self.data_frame = data_frame
        self.initial_data_frame = data_frame.copy()

    def data_cleaning(self):
        self.drop_row_if_has_null_column()
        if not self.column_name:
            self.remove_column_from_data_frame()
        self.sanitize_data_frame()
        if self.dataframe_name != 'request':
            self.compare_dataframes()
            self.create_new_dataframes()
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
            print('#### START ' + str(index) + ' ####')
            print('1. Text: ' + self.text)
            self.text = self.covert_text_to_lower_case()
            print('2. covert_text_to_lower_case: ' + self.text)
            self.text = self.expand_contractions()
            print('3. expand_contractions: ' + self.text)
            self.text = self.replace_slang_word()
            print('4. replace_slang_word: ' + self.text)
            self.text = self.remove_urls()
            print('5. remove_urls: ' + self.text)
            self.text = self.remove_html_tags()
            print('6. remove_html_tags: ' + self.text)
            self.text = self.remove_emojis()
            print('7. remove_emojis: ' + self.text)
            # self.text = self.remove_stopwords()
            # print('6. remove_stopwords: ' + self.text)
            self.text = self.remove_hash_tags()
            print('8. remove_hash_tags: ' + self.text)
            self.text = self.convert_accented_characters_to_ASCII_characters()
            print('9. remove_accented_chars: ' + self.text)
            self.text = self.remove_punctuations_special_characters()
            print('10. remove_punctuations_special_characters: ' + self.text)
            self.text = self.remove_consequently_char()
            print('11. remove_consequently_char: ' + self.text)
            self.text = self.replace_slang_word()
            print('12. replace_slang_word: ' + self.text)
            self.text = self.remove_numbers()
            print('13. remove_numbers: ' + self.text)
            self.text = self.trim_text()
            print('14. trim_text: ' + self.text)
            self.text = self.remove_double_spaces()
            print('15. remove_double_spaces: ' + self.text)
            self.text = self.auto_spelling()
            print('16. auto_spelling: ' + self.text)
            self.text = self.lemma()
            print('17. lemmatizing: ' + self.text)
            # self.text = self.stem()
            # print('15. stem: ' + self.text)
            print('#### STOP ' + str(index) + '#### \n')
            # print(row['text'] + ' --- ' + text)
            # TODO: remove ` ,
            self.data_frame.loc[index, 'text'] = self.text
        # print(self.data_frame)

    def remove_punctuations_special_characters(self):
        punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
        text_without_punctuations = ''
        for char in self.text:
            if char not in punctuations:
                text_without_punctuations = text_without_punctuations + char
            else:
                text_without_punctuations = text_without_punctuations + " "
        return text_without_punctuations

    def remove_html_tags(self):
        cleaner = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
        return re.sub(cleaner, '', self.text)

    def remove_urls(self):
        cleaner = re.compile('http\S+')
        return re.sub(cleaner, '', self.text)

    def remove_hash_tags(self):
        return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", self.text).split())

    def covert_text_to_lower_case(self):
        return self.text.lower()

    def trim_text(self):
        return self.text.strip()

    def remove_double_spaces(self):
        return re.sub('\s+', ' ', self.text)

    def remove_numbers(self):
        return re.sub(r'\d+', '', self.text)

    def convert_accented_characters_to_ASCII_characters(self):
        return unicodedata.normalize('NFKD', self.text).encode('ascii', 'ignore').decode('utf-8', 'ignore')

    def compare_dataframes(self):
        merge_dataframes = pd.concat([self.initial_data_frame['text'], self.data_frame['text']], axis=1,
                                     keys=['Initial Text', 'Cleaned Text'])
        merge_dataframes.to_csv('presentation/results/' + self.dataframe_name + "_dataframe_cleaned_initial.csv",
                                sep=',', encoding='utf-8', index=False)
        print(merge_dataframes)

    def create_new_dataframes(self):
        merge_dataframes = pd.concat([self.data_frame['text'], self.data_frame['sentiment']], axis=1,
                                     keys=['text', 'sentiment'])
        merge_dataframes.to_csv('datasets/cleaned/' + self.dataframe_name + "_dataframe_cleaned.csv",
                                sep=',', index=False, header=True)
        print(merge_dataframes)

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

    def remove_stopwords(self):
        stop_words = stopwords.words('english')
        return ' '.join([w for w in word_tokenize(self.text) if not w in stop_words])

    def stem(self):
        snowball_stemmer = SnowballStemmer('english')
        stemmed_word = [snowball_stemmer.stem(word) for sent in sent_tokenize(self.text) for word in
                        word_tokenize(sent)]
        return " ".join(stemmed_word)

    def lemma(self):
        wordnet_lemmatizer = WordNetLemmatizer()
        lemmatized_word = [wordnet_lemmatizer.lemmatize(word) for sent in sent_tokenize(self.text) for word in
                           word_tokenize(sent)]
        return " ".join(lemmatized_word)

    def word_tokenize(self):
        return [w for sent in sent_tokenize(self.text) for w in word_tokenize(sent)]
