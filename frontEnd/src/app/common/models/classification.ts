export class Classification {
    text: string;
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
}
