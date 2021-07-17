from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, accuracy_score, recall_score, f1_score, \
    precision_score, roc_curve, classification_report, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import configparser
import numpy as np
from sklearn.model_selection import KFold

from definitions import ROOT_DIR
from utils.functions import get_sentiment_as_array, sanitize_model_name

path = Path()

config = configparser.RawConfigParser()
config.read('ConfigFile.properties')


class ComposeMetrics:
    plt_show = data_set = config.get('PROJECT', 'plt.show')

    def __init__(self, model, x_train, y_train, y_score, y_test, y_predict, model_name, data_name, word_embedding):
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
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
        # self.measure_trade_off_based_on_error()
        # self.measure_trade_off_based_on_accuracy()

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
        plt.close()

    def save_plot_as_image(self, type_name):
        """Save the plot as image"""
        plt.savefig(ROOT_DIR + '/presentation/images/' + type_name + '/' + self.model_name_for_save +
                    '_' + self.data_name + '_' + self.word_embedding + '_' + type_name + '.png',
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
        plt.title(self.model_name + ' ROC curve')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive rate')
        plt.legend(loc='best')
        self.save_plot_as_image('roc_curve')
        self.plt_show: plt.show()
        plt.close()

    def measure_trade_off_based_on_error(self):
        y = np.array(self.y_train)
        kf = KFold(n_splits=10)
        list_training_error = []
        list_testing_error = []
        for train_index, test_index in kf.split(self.x_train):
            x_train, x_test = self.x_train[train_index], self.x_train[test_index]
            y_train, y_test = y[train_index], y[test_index]
            self.model.fit(x_train, y_train)
            y_train_data_predictions = self.model.predict(x_train)
            y_test_data_predictions = self.model.predict(x_test)
            fold_training_error = mean_absolute_error(y_train, y_train_data_predictions)
            fold_testing_error = mean_absolute_error(y_test, y_test_data_predictions)
            list_training_error.append(fold_training_error)
            list_testing_error.append(fold_testing_error)

        plt.subplot(1, 2, 1)
        plt.plot(range(1, kf.get_n_splits() + 1), np.array(list_training_error).ravel(), 'o-')
        plt.xlabel('Number of fold')
        plt.ylabel('Training error')
        plt.title(self.model_name + ': Training error across folds')
        plt.tight_layout()
        plt.subplot(1, 2, 2)
        plt.plot(range(1, kf.get_n_splits() + 1), np.array(list_testing_error).ravel(), 'o-')
        plt.xlabel('Number of fold')
        plt.ylabel('Testing error')
        plt.title(self.model_name + ': Testing error across folds')
        plt.tight_layout()
        self.save_plot_as_image('model_trade_off')
        self.plt_show: plt.show()
        plt.close()

    def measure_trade_off_based_on_accuracy(self):
        y = np.array(self.y_train)
        k_fold = KFold(n_splits=10)
        train_scores = []
        test_scores = []
        for train, test in k_fold.split(self.x_train):
            x_train, x_test = self.x_train[train], self.x_train[test]
            y_train, y_test = y[train], y[test]
            self.model.fit(x_train, y_train)
            train_score = self.model.score(x_train, y_train)
            test_score = self.model.score(x_test, y_test)
            train_scores.append(train_score)
            test_scores.append(test_score)

        plt.plot(train_scores, color='red', label='Training Accuracy')
        plt.plot(test_scores, color='blue', label='Testing Accuracy')
        plt.xlabel('K values')
        plt.ylabel('Accuracy Score')
        plt.title(self.model_name + ': Performance Under Varying K Values')
        plt.legend(loc='best')
        self.save_plot_as_image('model_trade_off_alternative')
        self.plt_show: plt.show()
        plt.close()
