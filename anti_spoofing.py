# -*- coding: utf-8 -*-
# @Time : 20-6-9 下午3:06
# @Author : zhuying
# @Company : Minivision
# @File : test.py
# @Software : PyCharm

import cv2
import numpy as np
import warnings
import time

from src.anti_spoof_predict import AntiSpoofPredict
warnings.filterwarnings('ignore')

device_id = 0
model_testv1 = AntiSpoofPredict(device_id)
model_testv1._load_model("resources/anti_spoof_models/4_0_0_80x80_MiniFASNetV1SE.pth")

model_testv2 = AntiSpoofPredict(device_id)
model_testv2._load_model("resources/anti_spoof_models/4_0_0_80x80_MiniFASNetV1SE.pth")

def anti_spoofing(img):
    img = cv2.resize(img, (80, 80))
    prediction = np.zeros((1, 3))
    test_speed = 0
    
    start = time.time()

    prediction += model_testv1.predict(img)
    prediction += model_testv2.predict(img)
    test_speed += time.time()-start
    
    # # draw result of prediction
    label = np.argmax(prediction)
    score = prediction[0][label]/2
    if label == 1:
        return False, score     # Image 'datasets/temp.jpg' is Real Face. Score: 0.74.
    else:
        return True, score      # Image 'datasets/temp.jpg' is Fake Face. Score: 0.68
    
# Run function one time when initial to load model.
img = cv2.imread("datasets/temp.jpg")
anti_spoofing(img)
