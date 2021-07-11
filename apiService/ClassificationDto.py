from utils.helper_functions.functions import map_sentiment


class ClassificationDto:

    def __init__(self, bag_of_words_cleaned_sample_data_frame, bag_of_words_logistic_regression_probabilities_results,
                 bag_of_words_logistic_regression_results, bag_of_words_svm_results, bag_of_words_nb_results,
                 bag_of_words_mlp_results, bag_of_words_decision_tree_results,
                 tfidf_logistic_regression_probabilities_results, tfidf_logistic_regression_results, tfidf_svm_results,
                 tfidf_nb_results, tfidf_mlp_results, tfidf_decision_tree_results,
                 word2vec_logistic_regression_model_results, word2vec_logistic_regression_probabilities_results,
                 word2vec_svm_results, word2vec_mlp_results, word2vec_decision_tree_results, data_pre_processing_steps):
        self.bag_of_words_cleaned_sample_data_frame = bag_of_words_cleaned_sample_data_frame
        self.bag_of_words_logistic_regression_probabilities_results = \
            bag_of_words_logistic_regression_probabilities_results
        self.bag_of_words_logistic_regression_results = bag_of_words_logistic_regression_results
        self.bag_of_words_svm_results = bag_of_words_svm_results
        self.bag_of_words_nb_results = bag_of_words_nb_results
        self.bag_of_words_mlp_results = bag_of_words_mlp_results
        self.bag_of_words_decision_tree_results = bag_of_words_decision_tree_results
        self.tfidf_logistic_regression_probabilities_results = tfidf_logistic_regression_probabilities_results
        self.word2vec_logistic_regression_probabilities_results = word2vec_logistic_regression_probabilities_results
        self.tfidf_logistic_regression_results = tfidf_logistic_regression_results
        self.tfidf_svm_results = tfidf_svm_results
        self.tfidf_nb_results = tfidf_nb_results
        self.tfidf_mlp_results = tfidf_mlp_results
        self.tfidf_decision_tree_results = tfidf_decision_tree_results
        self.word2vec_logistic_regression_model_results = word2vec_logistic_regression_model_results
        self.word2vec_svm_results = word2vec_svm_results
        self.word2vec_mlp_results = word2vec_mlp_results
        self.word2vec_decision_tree_results = word2vec_decision_tree_results
        self.data_pre_processing_steps = data_pre_processing_steps

    def get_response(self):
        text = self.bag_of_words_cleaned_sample_data_frame.iloc[0]['text']
        bag_of_words_lg = self.bag_of_words_logistic_regression_probabilities_results[0]
        word2vec_lg = self.word2vec_logistic_regression_probabilities_results[0]
        tfidf_lg = self.tfidf_logistic_regression_probabilities_results[0]

        return {
            'text': str(text),
            'data_pre_processing_steps': self.data_pre_processing_steps,
            'bag_of_words': {
                'logistic_regression_probabilities': {
                    'negative': str(bag_of_words_lg[0]),
                    'neutral': str(bag_of_words_lg[1]),
                    'positive': str(bag_of_words_lg[2])
                },
                'logistic_regression': {
                    'Sentiment': map_sentiment(self.bag_of_words_logistic_regression_results[0])
                },
                'svm': {
                    'Sentiment': map_sentiment(self.bag_of_words_svm_results[0])
                },
                'naive_bayes': {
                    'Sentiment': map_sentiment(self.bag_of_words_nb_results[0])
                },
                'mlp': {
                    'Sentiment': map_sentiment(self.bag_of_words_mlp_results[0])
                },
                'decision_tree': {
                    'Sentiment': map_sentiment(self.bag_of_words_decision_tree_results[0])
                }
            },
            'tfidf': {
                'logistic_regression_probabilities': {
                    'negative': str(tfidf_lg[0]),
                    'neutral': str(tfidf_lg[1]),
                    'positive': str(tfidf_lg[2])
                },
                'logistic_regression': {
                    'Sentiment': map_sentiment(self.tfidf_logistic_regression_results[0])
                },
                'svm': {
                    'Sentiment': map_sentiment(self.tfidf_svm_results[0])
                },
                'naive_bayes': {
                    'Sentiment': map_sentiment(self.tfidf_nb_results[0])
                },
                'mlp': {
                    'Sentiment': map_sentiment(self.tfidf_mlp_results[0])
                },
                'decision_tree': {
                    'Sentiment': map_sentiment(self.tfidf_decision_tree_results[0])
                }
            },
            'word2Vec': {
                'logistic_regression_probabilities': {
                    'negative': str(word2vec_lg[0]),
                    'neutral': str(word2vec_lg[1]),
                    'positive': str(word2vec_lg[2])
                },
                'logistic_regression': {
                    'Sentiment': map_sentiment(self.word2vec_logistic_regression_model_results[0])
                },
                'svm': {
                    'Sentiment': map_sentiment(self.word2vec_svm_results[0])
                },
                'naive_bayes': {
                    'Sentiment': map_sentiment(self.tfidf_nb_results[0])
                },
                'mlp': {
                    'Sentiment': map_sentiment(self.word2vec_mlp_results[0])
                },
                'decision_tree': {
                    'Sentiment': map_sentiment(self.word2vec_decision_tree_results[0])
                }
            }

        }
