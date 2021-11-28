# -*- coding: utf-8 -*-
# @Time : 20-6-9 下午3:06
# @Author : zhuying
# @Company : Minivision
# @File : test.py
# @Software : PyCharm

import os
import cv2
import numpy as np
import argparse
import time

from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name

import warnings
warnings.filterwarnings('ignore')

gpu_id = 0
model_test = AntiSpoofPredict(gpu_id)
image_cropper = CropImage()

def anti_sproofing(img, model_dir="./resources/anti_spoof_models"):
    try:
        image_bbox = model_test.get_bbox(img)
    except AttributeError as E:
        print(E)
        print("Image is None, check input image and try again!")
    
    prediction = np.zeros((1, 3))
    test_speed = 0
    # sum the prediction from single model's result
    for model_name in os.listdir(model_dir):
        h_input, w_input, model_type, scale = parse_model_name(model_name)
        param = {
            "org_img": img,
            "bbox": image_bbox,
            "scale": scale,
            "out_w": w_input,
            "out_h": h_input,
            "crop": True,
        }
        if scale is None:
            param["crop"] = False
        # Crop for predict anti sproofinf
        crop_img = image_cropper.crop(**param)
        
        start = time.time()
        prediction += model_test.predict(crop_img, os.path.join(model_dir, model_name))
        test_speed += time.time()-start
        
    # draw result of prediction
    label = np.argmax(prediction)
    score = prediction[0][label]/2
    if label == 1:
        # Image 'frame' is Real Face. Score: 0.99.  color = (255, 0, 0)
        return False, score
    else:
        # Image 'frame' is Fake Face. Score: 0.82.  color = (0, 0, 255)
        return True, score

# def anti_sproofting_show_frame():
#     if label == 1:
#         print("Image '{}' is Real Face. Score: {:.2f}.".format(image_name, value))
#         result_text = "RealFace Score: {:.2f}".format(value)
#         color = (255, 0, 0)
#     else:
#         print("Image '{}' is Fake Face. Score: {:.2f}.".format(image_name, value))
#         result_text = "FakeFace Score: {:.2f}".format(value)
#         color = (0, 0, 255)
#     print("Prediction cost {:.2f} s".format(test_speed))
#     cv2.rectangle(
#         image,
#         (image_bbox[0], image_bbox[1]),
#         (image_bbox[0] + image_bbox[2], image_bbox[1] + image_bbox[3]),
#         color, 2)
#     cv2.putText(
#         image,
#         result_text,
#         (image_bbox[0], image_bbox[1] - 5),
#         cv2.FONT_HERSHEY_COMPLEX, 0.5*image.shape[0]/1024, color)


# img = cv2.imread("image2.jpg")
# start = time.time()
# print("Image is fake:", anti_sproofing(img))
# end = time.time()
# print("Time cost:", end-start)
