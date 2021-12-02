import sys
import os
sys.path.append(os.path.abspath(os.path.join('..')))
sys.path.append("output")
sys.path.append('../insightface/deploy')
sys.path.append('../insightface/src/common')

from mtcnn.mtcnn import MTCNN
from imutils import paths
import face_preprocess

import numpy as np
import cv2
import argparse

ap = argparse.ArgumentParser()

ap.add_argument("--video", default="E:/Timekeeping/Face Recognition with InsightFace/datasets/videos_input/hoan.mp4",
                help="Number of faces that camera will get")
ap.add_argument("--output", default="../datasets/train/Hoan",
                help="Path to faces output")

args = vars(ap.parse_args())

def get_faces_from_folder(folder_path="datasets/train/Hoann"):
    imagePaths = list(paths.list_images(folder_path))
    detector = MTCNN()
    
    for i, img_path in enumerate(imagePaths):    
        # Get all faces on current frame
        
    #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #     results = face_detection.process(image)
    #     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    #     h, w, c = image.shape
        
    #     if results.detections:
    #         for id, detection in enumerate(results.detections):
    #             mp_draw.draw_detection(image, detection)
    #             bBox = detection.location_data.relative_bounding_box
    #             boundBox = int(bBox.xmin * w), int(bBox.ymin * h), int(bBox.width * w), int(bBox.height * h)

        img = cv2.imwrite(img_path)
        print(img_path)
        bboxes = detector.detect_faces(img)

        if len(bboxes) != 0:
            # Get only the biggest face
            for bboxe in bboxes:
                bbox = bboxe["box"]
                bbox = np.array([bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]])
                landmarks = bboxe["keypoints"]

                # convert to face_preprocess.preprocess input
                landmarks = np.array([landmarks["left_eye"][0], landmarks["right_eye"][0], landmarks["nose"][0], landmarks["mouth_left"][0], landmarks["mouth_right"][0],
                                    landmarks["left_eye"][1], landmarks["right_eye"][1], landmarks["nose"][1], landmarks["mouth_left"][1], landmarks["mouth_right"][1]])
                landmarks = landmarks.reshape((2,5)).T
                nimg = face_preprocess.preprocess(img, bbox, landmarks, image_size='112,112')
                cv2.imwrite(img, nimg)
                print(f"Preprocessed: {i+1} image")
                
                
get_faces_from_folder("datasets/train/Hoann")