from helper_functions.read_dataset import functions as read_dataset
from helper_functions.tokenize.TokenizeDataFrame import TokenizeDataFrame
from helper_functions.convert_test.functions import tokenizing_sentences_and_words, tokenizing_sentences
from helper_functions.text_vectorization.BoW import BoW
from helper_functions.text_vectorization.Word2VecModel import Word2VecModel

# Retrieve Data Frames
sample_data_frame = read_dataset.read_cleaned_sample_data_set()
train_data_frame = read_dataset.read_cleaned_train_data_set()
test_data_frame = read_dataset.read_cleaned_test_data_set()

# TODO: check nulls
train_data_frame.dropna(inplace=True)
test_data_frame.dropna(inplace=True)

# Tokenize data frame
tokenize_data_frame_sample_data = TokenizeDataFrame(sample_data_frame)
sample_corpus = tokenize_data_frame_sample_data.tokenize_sentence()

tokenize_data_frame_train_data = TokenizeDataFrame(train_data_frame)
train_corpus = tokenize_data_frame_train_data.tokenize_sentence()

tokenize_data_frame_test_data = TokenizeDataFrame(test_data_frame)
test_corpus = tokenize_data_frame_test_data.tokenize_sentence()

# Convert Text
sample_tokenized_sentences_data_frame = tokenizing_sentences_and_words(sample_data_frame)
sample_tokenized_sentences_data_frames = tokenizing_sentences(sample_data_frame)

# Vectorized - BOW
bag_of_words = BoW(sample_tokenized_sentences_data_frames, sample_corpus)
array = bag_of_words.vectorize_text()

# Vectorized - Word2Vec
word_2_vec = Word2VecModel(sample_tokenized_sentences_data_frame)
word_2_vec_model = word_2_vec.vectorize_text()
