import numpy as np
from data_upload_test import data_import_test
from data_upload_train import data_import_train

def normalization_train():

    img, label = data_import_train()
    img = np.array(img)
    label = np.array(label)
    img_normalized = img.astype('float32') / 255.0
    img_ready = img_normalized.reshape(-1, 28, 28, 1)

    return img_ready, label

def normalization_test():

    img, label = data_import_test()
    img = np.array(img)
    label = np.array(label)
    img_normalized = img.astype('float32') / 255.0
    img_ready = img_normalized.reshape(-1, 28, 28, 1)

    return img_ready, label