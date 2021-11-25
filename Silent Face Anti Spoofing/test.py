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
import warnings
import time

from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name
warnings.filterwarnings('ignore')


SAMPLE_IMAGE_PATH = "./images/sample/"


# 因为安卓端APK获取的视频流宽高比为3:4,为了与之一致，所以将宽高比限制为3:4
def check_image(image):
    height, width, channel = image.shape
    if width/height != 3/4:
        print("Image is not appropriate!!!\nHeight/Width should be 4/3.")
        return False
    else:
        return True

def resize_to_43(img, aspect_ratio):
    try:
        imagex = int(img.size[0])
        imagey = int(img.size[1])
        width = min(imagex, imagey*aspect_ratio)
        height = min(imagex/aspect_ratio, imagey)
        left =(imagex - width)/2
        top = (imagey - height)/2
        box = (left,top,left+width,top+height)
        img = img.crop(box)
    except Exception as e:
        print(e)
        pass
    return img

# def test(image_name, model_dir, device_id):
#     model_test = AntiSpoofPredict(device_id)
#     image_cropper = CropImage()
#     image = cv2.imread(SAMPLE_IMAGE_PATH + image_name)

def test(video_name, model_dir, device_id):
    # cap = cv2.VideoCapture(video_name)
    # if (cap.isOpened()== False): 
    #     print("Error opening video  file")
        
    # while(cap.isOpened()):
    #     ret, frame = cap.read()
    #     if ret == True:
    #         cv2.imshow('frame', frame)
    #         if cv2.waitKey(1) & 0xFF == ord('q') or ret==False :
    #             break
    #     else: 
    #         break
    
    model_test = AntiSpoofPredict(device_id)
    image_cropper = CropImage()
    cap = cv2.VideoCapture(video_name)
    
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            if cv2.waitKey(1) & 0xFF == ord('q') or ret==False :
                break
            
            image_name = "frame"
            result = check_image(frame)
            if result is False:
                # frame = resize_to_43(frame, 3/4)
                return
            
            image_bbox = model_test.get_bbox(frame)
            prediction = np.zeros((1, 3))
            test_speed = 0
            # sum the prediction from single model's result
            for model_name in os.listdir(model_dir):
                h_input, w_input, model_type, scale = parse_model_name(model_name)
                param = {
                    "org_img": frame,
                    "bbox": image_bbox,
                    "scale": scale,
                    "out_w": w_input,
                    "out_h": h_input,
                    "crop": True,
                }
                if scale is None:
                    param["crop"] = False
                img = image_cropper.crop(**param)
                start = time.time()
                prediction += model_test.predict(img, os.path.join(model_dir, model_name))
                test_speed += time.time()-start

            # draw result of prediction
            label = np.argmax(prediction)
            value = prediction[0][label]/2
            if label == 1:
                print("Image '{}' is Real Face. Score: {:.2f}.".format(image_name, value))
                result_text = "RealFace Score: {:.2f}".format(value)
                color = (255, 0, 0)
            else:
                print("Image '{}' is Fake Face. Score: {:.2f}.".format(image_name, value))
                result_text = "FakeFace Score: {:.2f}".format(value)
                color = (0, 0, 255)
            print("Prediction cost {:.2f} s".format(test_speed))
            cv2.rectangle(
                frame,
                (image_bbox[0], image_bbox[1]),
                (image_bbox[0] + image_bbox[2], image_bbox[1] + image_bbox[3]),
                color, 2)
            cv2.putText(
                frame,
                result_text,
                (image_bbox[0], image_bbox[1] - 5),
                cv2.FONT_HERSHEY_COMPLEX, 0.5*frame.shape[0]/1024, color)
            
            cv2.imshow('Frame', frame)
            # format_ = os.path.splitext(image_name)[-1]
            # result_image_name = image_name.replace(format_, "_result" + format_)
            # cv2.imwrite(SAMPLE_IMAGE_PATH + result_image_name, frame)
        
        else: 
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    desc = "test"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        "--device_id",
        type=int,
        default=0,
        help="which gpu id, [0/1/2/3]")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="./resources/anti_spoof_models",
        help="model_lib used to test")
    parser.add_argument(
        "--video_input",
        type=str,
        default="E:/Timekeeping/Silent-Face-Anti-Spoofing-master/images/sample/facenliveness.mp4",
        help="image used to test")
    args = parser.parse_args()
    test(args.video_input, args.model_dir, args.device_id)
    