# -*- coding: utf-8 -*-
# @Time : 20-6-9 下午3:06
# @Author : zhuying
# @Company : Minivision
# @File : test.py
# @Software : PyCharm

import cv2
import numpy as np
import time
import logging

from src.anti_spoof_predict import AntiSpoofPredict

# Init logger
logger = logging.getLogger(__name__)
logger_file = logging.FileHandler('logging.log')
logger_format = logging.Formatter("%(asctime)s::%(levelname)s::\t%(filename)s::%(funcName)s()::%(lineno)d\t::%(message)s"
                                  , datefmt="%Y-%m-%d %H:%M:%S")

logger_file.setFormatter(logger_format)
logger.addHandler(logger_file)
logger.setLevel(logging.INFO)

device_id = 0
modelv1 = AntiSpoofPredict(device_id)
modelv1._load_model("resources/anti_spoof_models/4_0_0_80x80_MiniFASNetV1SE.pth")
logger.info("Load AntiSpoof v1 model")

modelv2 = AntiSpoofPredict(device_id)
modelv2._load_model("resources/anti_spoof_models/2.7_80x80_MiniFASNetV2.pth")
logger.info("Load AntiSpoof v2 model")

def anti_spoofing(img):
    img = cv2.resize(img, (80, 80))
    prediction = np.zeros((1, 3))
    test_speed = 0
    
    start = time.time()

    prediction += modelv1.predict(img)
    prediction += modelv2.predict(img)
    test_speed += time.time()-start
    
    # # draw result of prediction
    label = np.argmax(prediction)
    score = prediction[0][label]/2
    if label == 1:
        return False, score     # Image 'datasets/temp.jpg' is Real Face. Score: 0.74.
    else:
        return True, score      # Image 'datasets/temp.jpg' is Fake Face. Score: 0.68
    
# Run function one time when initial to load model.
logger.info("Run function one time when initial to load model with datasets/temp.jpg image")
img = cv2.imread("datasets/temp.jpg")
anti_spoofing(img)
