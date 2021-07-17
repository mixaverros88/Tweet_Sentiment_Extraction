from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer
from num2words import num2words
from autocorrect import Speller
from wordsegment import load, segment
from pathlib import Path
import unicodedata
import pandas as pd
import re
import nltk
import en_core_web_sm
from .contractionList import contractions_list
from .slanglist import slang_list

nlp = en_core_web_sm.load()

path = Path()
nltk.download('maxent_ne_chunker')
nltk.download('words')


class DataCleaning:
    data_pre_processing_steps = {}
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
            self.create_new_csv_with_cleaned_text()
            self.create_new_csv_for_comparison()
        return self.data_frame

    def drop_row_if_has_null_column(self):
        """Drop All Rows with any Null/NaN/NaT Values"""
        self.data_frame.dropna(inplace=True)

    def remove_column_from_data_frame(self):
        """Remove a column from give data frame """
        self.data_frame.drop(self.column_name, axis=1, inplace=True)

    def get_data_pre_processing_steps(self):
        return self.data_pre_processing_steps

    def sanitize_data_frame(self):
        """For every row of the data frame proceed with the following steps"""
        for index, row in self.data_frame.iterrows():
            self.text = row['text']
            print('#### START: ' + str(index) + ' ####')
            print('1. Initial Text: ' + self.text)
            self.data_pre_processing_steps.update({'Step_01': self.text})
            self.text = self.name_entity_recognition()
            print('2. Name Entity Recognition: ' + self.text)
            self.data_pre_processing_steps.update({'Step_02': self.text})
            self.text = self.expand_contractions()
            print('3. Expand Contractions: ' + self.text)
            self.data_pre_processing_steps.update({'Step_03': self.text})
            self.text = self.remove_am_pm_dates()
            print('4. Remove am pm dates: ' + self.text)
            self.data_pre_processing_steps.update({'Step_04': self.text})
            self.text = self.remove_emojis()
            print('5. Remove Emojis: ' + self.text)
            self.data_pre_processing_steps.update({'Step_05': self.text})
            self.text = self.covert_text_to_lower_case()
            print('6. Covert Text To Lower Case: ' + self.text)
            self.data_pre_processing_steps.update({'Step_06': self.text})
            self.text = self.replace_slang_word()
            print('7. Replace Slang Word: ' + self.text)
            self.data_pre_processing_steps.update({'Step_07': self.text})
            self.text = self.remove_urls()
            print('8. Remove Urls: ' + self.text)
            self.data_pre_processing_steps.update({'Step_08': self.text})
            self.text = self.remove_html_tags()
            print('9. Remove Html Tags: ' + self.text)
            self.data_pre_processing_steps.update({'Step_09': self.text})
            self.text = self.remove_hash_tags()
            print('10. Remove Hash Tags: ' + self.text)
            self.data_pre_processing_steps.update({'Step_10': self.text})
            self.text = self.convert_accented_characters_to_ASCII_characters()
            print('11. Remove Accented Chars: ' + self.text)
            self.data_pre_processing_steps.update({'Step_11': self.text})
            self.text = self.remove_punctuations_special_characters()
            print('12. Remove Punctuations Special Characters: ' + self.text)
            self.data_pre_processing_steps.update({'Step_12': self.text})
            self.text = self.remove_consequently_char()
            print('13. Remove Consequently Char: ' + self.text)
            self.data_pre_processing_steps.update({'Step_13': self.text})
            self.text = self.replace_slang_word()
            print('14. Replace Slang Word: ' + self.text)
            self.data_pre_processing_steps.update({'Step_14': self.text})
            self.text = self.trim_text()
            print('15. Trim Text: ' + self.text)
            self.data_pre_processing_steps.update({'Step_15': self.text})
            self.text = self.remove_double_spaces()
            print('16. Remove Double Spaces: ' + self.text)
            self.data_pre_processing_steps.update({'Step_16': self.text})
            self.text = self.word_segment()
            print('17. Word Segment: ' + self.text)
            self.data_pre_processing_steps.update({'Step_17': self.text})
            self.text = self.auto_spelling()
            print('18. Auto Spelling: ' + self.text)
            self.data_pre_processing_steps.update({'Step_18': self.text})
            self.text = self.lemmatization()
            print('19. Lemmatization: ' + self.text)
            self.data_pre_processing_steps.update({'Step_19': self.text})
            self.text = self.convert_number_to_word()
            print('20. Convert Num To Word: ' + self.text)
            self.data_pre_processing_steps.update({'Step_20': self.text})
            self.text = self.convert_to_nominal()
            print('21. Convert To Nominal: ' + self.text)
            self.data_pre_processing_steps.update({'Step_21': self.text})
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

    def create_new_csv_with_cleaned_text(self):
        """Create a new csv with cleaned text for fitting the models"""
        merge_dataframes = pd.concat([self.data_frame['text'], self.data_frame['sentiment']], axis=1,
                                     keys=['text', 'sentiment'])
        merge_dataframes.to_csv('../datasets/cleaned/' + self.dataframe_name + "_dataframe_cleaned.csv",
                                sep=',', index=False, header=True)

    def create_new_csv_for_comparison(self):
        """ Summarize the raw tweet with the cleaned tweet in one csv in order to analyze the results"""
        merge_dataframes = pd.concat([self.initial_data_frame['text'], self.data_frame['text']], axis=1,
                                     keys=['Initial Text', 'Cleaned Text'])
        merge_dataframes.to_csv('../presentation/results/' + self.dataframe_name + "_dataframe_cleaned_initial.csv",
                                sep=',', encoding='utf-8', index=False)

    def remove_emojis(self):
        """ Remove Emoji e.g. 👋 """
        regex_pattern = \
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
        return regex_pattern.sub(r'', self.text)

    def expand_contractions(self):
        """Expand contractions e.g. ain't -> am not"""
        for word in self.text.split():
            if word.lower() in contractions_list:
                self.text = self.text.replace(word, contractions_list[word.lower()])
        return self.text

    def remove_consequently_char(self):
        """Remove Consequently Char e.g. ggggooooooddd -> good"""
        return re.sub(r'(.)\1+', r'\1\1', self.text)

    def replace_slang_word(self):
        """Replace Slang Word e.g. afaik -> as far as i know"""
        for word in self.text.split():
            if word.lower() in slang_list:
                self.text = self.text.replace(word, slang_list[word.lower()])
        return self.text

    def auto_spelling(self):
        """Replace mis spelling words e.g. godi -> god"""
        spell = Speller(lang='en')
        spells = [spell(w) for w in (word_tokenize(self.text))]
        return " ".join(spells)

    # https://www.datacamp.com/community/tutorials/stemming-lemmatization-python
    def lemmatization(self):
        """Get the lemma for every word e.g. running -> run"""
        wordnet_lemmatizer = WordNetLemmatizer()
        lemmatized_word = [wordnet_lemmatizer.lemmatize(word, pos="v") for sent in sent_tokenize(self.text) for word in
                           word_tokenize(sent)]
        return " ".join(lemmatized_word)

    def convert_number_to_word(self):
        """Convert number to word e.g 12 to twelve"""
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
        new_text = self.text
        for n in numbers:
            ordinal_as_string = num2words(n, ordinal=True)
            new_text = re.sub(r"\d+[st|nd|rd|th]", ordinal_as_string[:-1], self.text, 1)
        print(self.text)
        print(new_text)
        return new_text

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
            # REMOVE organizations, Person, Location, Time
            if X.label_ == 'ORG' or \
                    X.label_ == 'PERSON' or \
                    X.label_ == 'GPE' or \
                    X.label_ == 'TIME':
                self.text = self.text.replace(X.text, '')
        print(self.text)
        return self.text
