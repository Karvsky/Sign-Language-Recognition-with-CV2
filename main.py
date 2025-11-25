import numpy as np
import cv2
from data_normalization import normalization_train, normalization_test
from model import cnns_model
from data_upload_test import data_import_test
from data_upload_train import data_import_train

#cap = cv2.VideoCapture(0)

#while True:
    #sucess,img = cap.read()
    #cv2.imshow("Video", img)

    #if cv2.waitKey(1) & 0xFF == ord('q'):
        #break

img_train, labels_train = data_import_train()
img_test, labels_test = data_import_test()

images_normalized_train = normalization_train()
image_normalized_test = normalization_test()

model = cnns_model()

model.compile(optimezer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(images_normalized_train, labels_train, epochs=15, validation_data=(image_normalized_test, labels_test))

loss, accuracy = model.evaluate(image_normalized_test, labels_test, verbose=1)




