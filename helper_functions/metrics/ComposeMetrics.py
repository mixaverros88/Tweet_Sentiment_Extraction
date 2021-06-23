from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, accuracy_score, recall_score, f1_score, \
    precision_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


class ComposeMetrics:

    def __init__(self, y_test, y_predict, model_name, **targets_name_array):
        self.y_test = y_test
        self.y_predict = y_predict
        self.model_name = model_name
        self.targets_name_array = [0, 1, 2]
        self.print_macro_metrics()
        self.plot_confusion_matrix()

    def print_macro_metrics(self):
        print(self.model_name + ' : Macro Precision, recall, f1-score',
              precision_recall_fscore_support(self.y_test, self.y_predict, average='macro'))
        print(self.model_name + ' : Micro Precision, recall, f1-score',
              precision_recall_fscore_support(self.y_test, self.y_predict, average='micro'))

    def print_metrics(self):
        print(self.model_name + ' Accuracy: ', round(accuracy_score(self.y_test, self.y_pred), 2), '%')
        print(self.model_name + ' Recall: ', round(recall_score(self.y_test, self.y_pred), 2), '%')
        print(self.model_name + ' Precision: ', round(precision_score(self.y_test, self.y_pred), 2), '%')
        print(self.model_name + ' F-measure: ', round(f1_score(self.y_test, self.y_pred), 2), '%')
        print(self.model_name + ' Confusion Matrix: \n', confusion_matrix(self.y_test, self.y_pred))

    def plot_confusion_matrix(self):
        confusion_matrix_array = confusion_matrix(self.y_test, self.y_predict)
        confusion_matrix_data_frame = pd.DataFrame(confusion_matrix_array, index=self.targets_name_array,
                                                   columns=self.targets_name_array)
        plt.figure(figsize=(5.5, 4))
        sns.heatmap(confusion_matrix_data_frame, annot=True)
        plt.title(self.model_name + ' Confusion Matrix')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()
