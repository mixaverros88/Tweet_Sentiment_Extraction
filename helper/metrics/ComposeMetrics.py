from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, accuracy_score, recall_score, f1_score, \
    precision_score, roc_curve, auc, classification_report
from helper.helper_functions.functions import get_sentiment_as_array, sanitize_model_name
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import os
import configparser

path = Path()

config = configparser.RawConfigParser()
config.read('ConfigFile.properties')


class ComposeMetrics:
    plt_show = data_set = config.get('PROJECT', 'plt.show')

    def __init__(self, y_score, y_test, y_predict, model_name, data_name, word_embedding):
        self.y_score = y_score
        self.y_test = y_test
        self.y_predict = y_predict
        self.model_name = model_name
        self.data_name = data_name
        self.word_embedding = word_embedding
        self.model_name_for_save = sanitize_model_name(model_name)
        self.targets_name_array = get_sentiment_as_array()
        self.print_micro_macro_metrics()
        self.plot_confusion_matrix()
        self.classification_report()
        if self.y_score is not None:
            self.plot_roc_curve()

    def print_micro_macro_metrics(self):
        print(self.model_name + ' : Macro Precision, recall, f1-score',
              precision_recall_fscore_support(self.y_test, self.y_predict, average='macro'))
        print(self.model_name + ' : Micro Precision, recall, f1-score',
              precision_recall_fscore_support(self.y_test, self.y_predict, average='micro'))

    def classification_report(self):
        print(classification_report(self.y_test, self.y_predict))

    def print_metrics(self):
        print(self.model_name + ' Accuracy: ', round(accuracy_score(self.y_test, self.y_predict), 2), '%')
        print(self.model_name + ' Recall: ', round(recall_score(self.y_test, self.y_predict), 2), '%')
        print(self.model_name + ' Precision: ', round(precision_score(self.y_test, self.y_predict), 2), '%')
        print(self.model_name + ' F-measure: ', round(f1_score(self.y_test, self.y_predict), 2), '%')
        print(self.model_name + ' Confusion Matrix: \n', confusion_matrix(self.y_test, self.y_predict))

    def plot_confusion_matrix(self):
        confusion_matrix_array = confusion_matrix(self.y_test, self.y_predict)
        print(confusion_matrix_array)
        confusion_matrix_data_frame = pd.DataFrame(
            confusion_matrix_array,
            index=self.targets_name_array,
            columns=self.targets_name_array)
        plt.figure(figsize=(10, 7))
        sns.heatmap(confusion_matrix_data_frame, annot=True, fmt="d", cmap='Blues')
        plt.title(self.model_name + ' Confusion Matrix')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        self.save_plot_as_image('confusion_matrix')
        self.plt_show: plt.show()

    def save_plot_as_image(self, type_name):
        """Save the plot as image"""
        plt.savefig(os.path.abspath(
            __file__ + '/../../../presentation/images/' + type_name + '/' + self.model_name_for_save +
            '_' + self.data_name + '_' + self.word_embedding + '_' + type_name + '.png'),
            bbox_inches='tight')

    def plot_roc_curve(self):
        # roc_curve curve for classes
        fpr = {}
        tpr = {}
        thresh = {}

        n_class = 3

        for i in range(n_class):
            fpr[i], tpr[i], thresh[i] = roc_curve(self.y_test, self.y_score[:, i], pos_label=i)

        # plotting
        plt.plot(fpr[0], tpr[0], linestyle='--', color='orange', label='Negative')
        plt.plot(fpr[1], tpr[1], linestyle='--', color='green', label='Neutral')
        plt.plot(fpr[2], tpr[2], linestyle='--', color='blue', label='Positive')
        plt.title('Multiclass ROC curve')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive rate')
        plt.legend(loc='best')
        self.save_plot_as_image('roc_curve')
        self.plt_show: plt.show()
