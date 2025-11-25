import numpy as np
import pandas as pd


def data_import_test():
    dataset = pd.read_csv('C:\\Users\\Karol\\PycharmProjects\\Sign language recognition\\Dataset\\sign_mnist_test.csv')
    pixels = dataset.drop(columns=['label'], axis=1).values
    labels = dataset['label'].values

    images = pixels.reshape(-1,28,28)

    return images, labels