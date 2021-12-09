# -*- coding: utf-8 -*-
# @Time : 20-6-9 下午3:06
# @Author : zhuying
# @Company : Minivision
# @File : test.py
# @Software : PyCharm

import os
import numpy as np
import time
import cv2

from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name

import warnings
warnings.filterwarnings('error')

gpu_id = 0
model_testv1 = AntiSpoofPredict(gpu_id)
model_testv2 = AntiSpoofPredict(gpu_id)
model_testv1._load_model("resources/anti_spoof_models/4_0_0_80x80_MiniFASNetV1SE.pth")
model_testv2._load_model("resources/anti_spoof_models/2.7_80x80_MiniFASNetV2.pth")

image_cropper = CropImage()

def anti_sproofing(img, image_bbox=None, model_dir="./resources/anti_spoof_models"):
    prediction = np.zeros((1, 3))
    if image_bbox is None:
        img = cv2.resize(img, (80, 80))
        
        prediction += model_testv1.predict(img)
        prediction += model_testv2.predict(img)
        
        # Draw result of prediction
        label = np.argmax(prediction)
        score = prediction[0][label]/2
        
        if label == 1 and score > 0.75:
            return False, score  # Image 'frame' is Real Face. Score: 0.99.  color = (255, 0, 0)
        else:
            return True, score # Image 'frame' is Fake Face. Score: 0.82.  color = (0, 0, 255)
            
    else:
        # sum the prediction from single model's result
        for model_name in os.listdir(model_dir):
            h_input, w_input, _, scale = parse_model_name(model_name)
            info = model_name.split('_')[0:-1]
            scale = float(info[0])
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
            
        prediction += model_testv1.predict(crop_img)
        prediction += model_testv2.predict(crop_img)
        
        # Draw result of prediction
        label = np.argmax(prediction)
        score = prediction[0][label]/2
        
        if label == 1 and score > 0.75:
            # Image 'frame' is Real Face. Score: 0.99.  color = (255, 0, 0)
            return False, score
        else:
            # Image 'frame' is Fake Face. Score: 0.82.  color = (0, 0, 255)
            return True, score

# Run an sample image to load model
img = cv2.imread("datasets/temp.jpg")
anti_sproofing(img)
