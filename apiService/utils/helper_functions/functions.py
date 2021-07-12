from nltk import word_tokenize
import pandas as pd


def convert_sentence_to_vector_array_request(word2vec_model, sentence):
    word_tokens = word_tokenize(sentence)
    np_vec = []
    for word in word_tokens:
        try:
            vec = word2vec_model.wv[str(word)]
            np_vec.append(vec)
        except:
            print(str(word) + ' is not in vocabulary word2vec')
    return np_vec


def map_sentiment(sentiment):
    if sentiment == 0:
        return 'Negative'
    if sentiment == 1:
        return 'Neutral'
    if sentiment == 2:
        return 'Positive'


def convert_text_to_data_frame_of_one_row(text):
    request_text = {'text': [text]}
    return pd.DataFrame(request_text)
