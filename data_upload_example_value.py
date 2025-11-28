import numpy as np
import pandas as pd
import cv2

def data_import_example_value():
    dataset = r'C:\Users\Karol\PycharmProjects\Sign language recognition\Dataset\test1.jpg'

    img = cv2.imread(dataset, cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img, (28, 28))
    img_normalized = img_resized.astype('float32') / 255.0
    img_final = img_normalized.reshape(1, 28, 28, 1)

    return img_final


