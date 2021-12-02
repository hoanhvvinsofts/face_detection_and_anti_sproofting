# -*- coding: utf-8 -*-
# @Time : 20-6-9 下午3:06
# @Author : zhuying
# @Company : Minivision
# @File : test.py
# @Software : PyCharm

import os
import numpy as np
import time

from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name

import warnings
warnings.filterwarnings('ignore')

gpu_id = 0
model_test = AntiSpoofPredict(gpu_id)
image_cropper = CropImage()

def anti_sproofing(img, image_bbox, model_dir="./resources/anti_spoof_models"):
    if image_bbox is None:
        prediction = np.zeros((1, 3))
        for model_name in os.listdir(model_dir):
            prediction += model_test.predict(img, os.path.join(model_dir, model_name))
        
        # draw result of prediction
        label = np.argmax(prediction)
        score = prediction[0][label]/2
        
        if label != 1 and score > 0.75:
            return True, score  # Image 'frame' is Real Face. Score: 0.99.  color = (255, 0, 0)
        else:
            return False, score # Image 'frame' is Fake Face. Score: 0.82.  color = (0, 0, 255)
            
    else:
        # start = time.time()
        # try:
        #     image_bbox = model_test.get_bbox(img)
        #     print("anti_sproofing(), image_bbox:", image_bbox)
        #     exit()
        # except AttributeError as E:
        #     print(E)
        #     print("Image is None, check input image and try again!")
        
        prediction = np.zeros((1, 3))
        # sum the prediction from single model's result
        for model_name in os.listdir(model_dir):
            h_input, w_input, _, scale = parse_model_name(model_name)
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
            prediction += model_test.predict(crop_img, os.path.join(model_dir, model_name))
            
        # draw result of prediction
        label = np.argmax(prediction)
        score = prediction[0][label]/2
        
        if label != 1 and score > 0.75:
            # Image 'frame' is Real Face. Score: 0.99.  color = (255, 0, 0)
            return True, score
        else:
            # Image 'frame' is Fake Face. Score: 0.82.  color = (0, 0, 255)
            return False, score

# img = cv2.imread("image2.jpg")
# start = time.time()
# print("Image is fake:", anti_sproofing(img))
# end = time.time()
# print("Time cost:", end-start)
