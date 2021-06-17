export class Classification {
    text: string;
    bag_of_words : {
      logistic_regression_proba :  {
        neutral: string;
        negative: string;
        positive: string;
      };
      logistic_regression : {
        Sentiment: string;
      };
      svm : {
        Sentiment: string;
      };
      naive_bayes : {
        Sentiment: string;
      };
      mlp : {
        Sentiment: string;
      };
      decision_tree : {
        Sentiment: string;
      };
    };
    word2Vec : {
      logistic_regression_proba :  {
        neutral: string;
        negative: string;
        positive: string;
      };
      logistic_regression : {
        Sentiment: string;
      };
      svm : {
        Sentiment: string;
      };
      naive_bayes : {
        Sentiment: string;
      };
      mlp : {
        Sentiment: string;
      };
      decision_tree : {
        Sentiment: string;
      };
    };
}
