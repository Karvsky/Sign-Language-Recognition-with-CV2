import numpy as np
import cv2
import random
import os
import tensorflow as tf
from data_normalization import normalization_train, normalization_test
from model import cnns_model
from data_upload_test import data_import_test
from data_upload_train import data_import_train
from data_upload_example_value import data_import_example_value

os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)


#cap = cv2.VideoCapture(0)

#while True:
    #sucess,img = cap.read()
    #cv2.imshow("Video", img)

    #if cv2.waitKey(1) & 0xFF == ord('q'):
        #break


img_train, labels_train = normalization_train()
img_test, labels_test = normalization_test()

model = cnns_model()

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(img_train, labels_train, epochs=3, validation_data=(img_test, labels_test))

loss, accuracy = model.evaluate(img_test, labels_test, verbose=1)

image = data_import_example_value()
print(image)
print(model.predict(image))



