export class Classification {
    text: string;
    data_pre_processing_steps : {
        Step_01: string;
        Step_02: string;
        Step_03: string;
        Step_04: string;
        Step_05: string;
        Step_06: string;
        Step_07: string;
        Step_08: string;
        Step_09: string;
        Step_10: string;
        Step_11: string;
        Step_12: string;
        Step_13: string;
        Step_14: string;
        Step_15: string;
        Step_16: string;
        Step_17: string;
        Step_18: string;
        Step_19: string;
        Step_20: string;
        Step_21: string;
    };
    bag_of_words : {
      logistic_regression_probabilities :  {
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
      logistic_regression_probabilities :  {
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
    tfidf : {
      logistic_regression_probabilities :  {
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
