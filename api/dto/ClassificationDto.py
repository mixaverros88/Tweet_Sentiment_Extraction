def map_sentiment(sentiment):
    if sentiment == 0:
        return 'Negative'
    if sentiment == 1:
        return 'Neutral'
    if sentiment == 2:
        return 'Positive'


class ClassificationDto:

    def __init__(self, cleaned_sample_data_frame, logistic_regression_probabilities_results,
                 logistic_regression_results, svm_results, nb_results, mlp_results, decision_tree_results):
        self.cleaned_sample_data_frame = cleaned_sample_data_frame
        self.logistic_regression_probabilities_results = logistic_regression_probabilities_results
        self.logistic_regression_results = logistic_regression_results
        self.svm_results = svm_results
        self.nb_results = nb_results
        self.mlp_results = mlp_results
        self.decision_tree_results = decision_tree_results

    def get_response(self):
        text = self.cleaned_sample_data_frame.iloc[0]['text']
        lg = self.logistic_regression_probabilities_results[0]

        return {
            'text': str(text),
            'bag_of_words': {
                'logistic_regression_probabilities': {
                    'negative': str(lg[0]),
                    'neutral': str(lg[1]),
                    'positive': str(lg[2])
                },
                'logistic_regression': {
                    'Sentiment': map_sentiment(self.logistic_regression_results[0])
                },
                'svm': {
                    'Sentiment': map_sentiment(self.svm_results[0])
                },
                'naive_bayes': {
                    'Sentiment': map_sentiment(self.nb_results[0])
                }
                ,
                'mlp': {
                    'Sentiment': map_sentiment(self.mlp_results[0])
                },
                'decision_tree': {
                    'Sentiment': map_sentiment(self.decision_tree_results[0])
                }
            },
            'word2Vec': {
                'logistic_regression_probabilities': {
                    'negative': str(lg[0]),
                    'neutral': str(lg[1]),
                    'positive': str(lg[2])
                },
                'logistic_regression': {
                    'Sentiment': map_sentiment(self.logistic_regression_results[0])
                },
                'svm': {
                    'Sentiment': map_sentiment(self.svm_results[0])
                },
                'naive_bayes': {
                    'Sentiment': map_sentiment(self.nb_results[0])
                }
                ,
                'mlp': {
                    'Sentiment': map_sentiment(self.mlp_results[0])
                },
                'decision_tree': {
                    'Sentiment': map_sentiment(self.decision_tree_results[0])
                }
            }

        }
