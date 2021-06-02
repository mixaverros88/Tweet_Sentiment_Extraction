from nltk import sent_tokenize

# TODO: Tokenizing sentences

def tokenizing_sentences(data_frame):
    data_frame['tokenized_sents'] = data_frame.apply(lambda row: sent_tokenize(row['text']), axis=1)
    return data_frame


# TODO: Normalizing words
def stemming_words():
    return


def lemmatization_words():
    return


# TODO: Vectorizing , Bag Of Words
def vectorizing_text():
    return
