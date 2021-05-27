import pandas as pd
import os

print(os.path.abspath(os.curdir))


def read_train_data_set():
    return pd.read_csv(os.path.abspath(os.curdir) + '\\datasets\\train.csv', sep=',')


def read_test_data_set():
    return pd.read_csv(os.path.abspath(os.curdir) + '\\test\\train.csv', sep=',')
