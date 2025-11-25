import numpy as np
import pandas as pd

dataset = pd.read_csv('C:\\Users\\Karol\\PycharmProjects\\Sign language recognition\\Dataset\\sign_mnist_train.csv')
pixels = dataset.drop(columns=['label'], axis=1).values
labels = dataset['label'].values

photos = pixels.reshape(-1,28,28)


