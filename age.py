#
# Credits to the model and for the code to https://github.com/yu4u/age-gender-estimation
# Just some changes to make it works as API service
#

import os
import cv2
import dlib
import numpy as np
import argparse
from contextlib import contextmanager
from wide_resnet import WideResNet
from keras.utils.data_utils import get_file
import imutils
from PIL import Image

pretrained_model = "https://github.com/yu4u/age-gender-estimation/releases/download/v0.5/weights.18-4.06.hdf5"
modhash = '89f56a39a78454e96379348bddd78c0d'


def AgeEstimator(uImage):
    depth = 16
    k = 8
    weight_file = get_file("weights.18-4.06.hdf5", pretrained_model, cache_subdir="pretrained_models",
                               file_hash=modhash, cache_dir=os.path.dirname(os.path.abspath(__file__)))
    detector = dlib.get_frontal_face_detector()
    img_size = 64
    model = WideResNet(img_size, depth=depth, k=k)()
    model.load_weights(weight_file)
    image = Image.open(uImage).convert('RGB')
    image = np.array(image)
    img_h, img_w, _ = np.shape(image)
    detected = detector(image, 1)
    faces = np.empty((len(detected), img_size, img_size, 3))
    print(faces)
    response = {}
    if len(detected) > 0:
        for i, d in enumerate(detected):
            x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + 1, d.bottom() + 1, d.width(), d.height()
            xw1 = max(int(x1 - 0.4 * w), 0)
            yw1 = max(int(y1 - 0.4 * h), 0)
            xw2 = min(int(x2 + 0.4 * w), img_w - 1)
            yw2 = min(int(y2 + 0.4 * h), img_h - 1)
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            faces[i, :, :, :] = cv2.resize(image[yw1:yw2 + 1, xw1:xw2 + 1, :], (img_size, img_size))

        results = model.predict(faces)
        predicted_genders = results[0]
        ages = np.arange(0, 101).reshape(101, 1)
        predicted_ages = results[1].dot(ages).flatten()
        for i, d in enumerate(detected):
            response[i] = {}
            response[i]["predicted_ages"] = "{}".format(int(predicted_ages[i]))
            response[i]["predicted_genders"] = "{}".format("F" if predicted_genders[i][0] > 0.5 else "M")
    return response
