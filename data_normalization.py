from data_upload_test import data_import
from data_upload_train import data_import

def normalization_train():

    img, label = data_import()
    normalized_dataset = []

    for i in img:
        img_float = i.astype('float32')
        normalized_dataset.append(img_float / 255.0)

    return normalized_dataset

def normalization_test():

    img, label = data_import()
    normalized_dataset = []

    for i in img:
        img_float = i.astype('float32')
        normalized_dataset.append(img_float / 255.0)

    return normalized_dataset