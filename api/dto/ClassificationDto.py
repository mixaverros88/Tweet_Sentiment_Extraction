class ClassificationDto:

    def __init__(self, cleaned_sample_data_frame, logistic_regression_proba_results, logistic_regression_results,
                 svm_results, nb_results, mlp_results):
        self.cleaned_sample_data_frame = cleaned_sample_data_frame
        self.logistic_regression_proba_results = logistic_regression_proba_results
        self.logistic_regression_results = logistic_regression_results
        self.svm_results = svm_results
        self.nb_results = nb_results
        self.mlp_results = mlp_results

    def get_cleaned_text(self):
        return self.cleaned_sample_data_frame.iloc[0]['text']

    def get_array(self):
        return self.logistic_regression_proba_results

    def get_sentiment(self, sentiment):
        if sentiment == 0:
            return 'Negative'
        if sentiment == 1:
            return 'Neutral'
        if sentiment == 2:
            return 'Positive'

    def get_response(self):
        text = self.cleaned_sample_data_frame.iloc[0]['text']
        lg = self.logistic_regression_proba_results[0]

        return {
            'text': str(text),
            'logistic_regression_proba': {
                'negative': str(lg[0]),
                'neutral': str(lg[1]),
                'positive': str(lg[2])
            },
            'logistic_regression': {
                'Sentiment': self.get_sentiment(self.logistic_regression_results[0])
            },
            'svm': {
                'Sentiment': self.get_sentiment(self.svm_results[0])
            },
            'naive_bayes': {
                'Sentiment': self.get_sentiment(self.nb_results[0])
            }
            ,
            'mlp': {
                'Sentiment': self.get_sentiment(self.mlp_results[0])
            }

        }
