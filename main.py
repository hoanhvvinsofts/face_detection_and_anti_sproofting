from anti_spoofing import anti_spoofing
from faces_embedding import add_embedding
from train_model import train_svm

import mediapipe as mp
import numpy as np
import cv2
import torch
import os
import time
import shutil
import _thread
import sys
import pickle
import dlib

sys.path.append('insightface/deploy')
sys.path.append('insightface/src/common')

mp_facedetector = mp.solutions.face_detection
mp_draw = mp.solutions.drawing_utils
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

"""
Argparse for FACE_MODEL:EMBEDDING_MODEL
"""
import face_model
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--mymodel", default="src/outputs/my_model.h5",
    help="Path to recognizer model")
ap.add_argument("--le", default="src/outputs/le.pickle",
    help="Path to label encoder")
ap.add_argument("--embeddings", default="src/outputs/embeddings.pickle",
    help='Path to embeddings')
ap.add_argument("--video-out", default="datasets/videos_output/stream_test.mp4",
    help='Path to output video')
ap.add_argument('--image-size', default='112,112', help='')
ap.add_argument('--model', default='insightface/models/model-y1-test2/model,0', help='path to load model.')
ap.add_argument('--ga-model', default='', help='path to load model.')
ap.add_argument('--gpu', default=0, type=int, help='gpu id')
ap.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from begining')
ap.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
ap.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')
args = ap.parse_args()
embedding_model = face_model.FaceModel(args)

# Load SVM model
with open('src/outputs/model.pkl', 'rb') as f:
    model = pickle.load(f)

data = pickle.loads(open(args.embeddings, "rb").read())
le = pickle.loads(open(args.le, "rb").read())
embeddings = np.array(data['embeddings'])
labels = le.fit_transform(data['names'])

from src.anti_spoof_predict import AntiSpoofPredict
gpu_id = 0
model_testv1 = AntiSpoofPredict(gpu_id)
model_testv2 = AntiSpoofPredict(gpu_id)
model_testv1._load_model("resources/anti_spoof_models/4_0_0_80x80_MiniFASNetV1SE.pth")
model_testv2._load_model("resources/anti_spoof_models/2.7_80x80_MiniFASNetV2.pth")

def findCosineDistance(vector1, vector2):
    """
    Calculate cosine distance between two vector
    """
    vec1 = vector1.flatten()
    vec2 = vector2.flatten()

    a = np.dot(vec1.T, vec2)
    b = np.dot(vec1.T, vec1)
    c = np.dot(vec2.T, vec2)
    return 1 - (a/(np.sqrt(b)*np.sqrt(c)))

def CosineSimilarity(test_vec, source_vecs):
    """
    Verify the similarity of one vector to group vectors of one class
    """
    cos_dist = 0
    for source_vec in source_vecs:
        cos_dist += findCosineDistance(test_vec, source_vec)
    return cos_dist/len(source_vecs)

def import_label_and_train(unknow_folder="datasets/unlabel/unknown"):
    global model
    global data
    global le
    global embeddings
    global labels
    shutil.rmtree(unknow_folder, ignore_errors=True)
    if os.path.exists(unknow_folder) is False:
        os.makedirs(unknow_folder)
    time.sleep(2.5)
    input_frame = input(">> Input label name: ")
    target_folder = "datasets/train/" + input_frame
    if os.path.isdir(target_folder):
        print("This label has already exist, saved frames are not move to train folder! Try other label!")
    else:
        os.makedirs(target_folder)
        for img_path in os.listdir(unknow_folder):
            src = unknow_folder + "/" + img_path
            dst = target_folder + "/" + img_path
            try:
                os.rename(src, dst)
            except FileExistsError:
                print("WARNING: Duplicate file. Force overwrite file:", dst)
                os.remove(dst)
                os.rename(src, dst)
                pass
            
        print(">> New label add to Train data!")
        print(">> Start training. . .")
        print(">> 1 - Embedding faces. . .")
        add_embedding(input_frame, target_folder)
        print(">> 2 - Training new model. . .")
        train_svm()
        print(">> 3 - Reloading new model. . .")
        with open('src/outputs/model.pkl', 'rb') as f:
            model = pickle.load(f)
            
        data = pickle.loads(open(args.embeddings, "rb").read())
        le = pickle.loads(open(args.le, "rb").read())
        embeddings = np.array(data['embeddings'])
        labels = le.fit_transform(data['names'])
        print(">> Loaded new model, new faces can now recognizable!")

def check_and_save_image(nimg, folder="datasets/unlabel/unknown"):
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    count_path = 0
    while True:
        frame_file = folder + f"/unknown{count_path}.jpg"
        if os.path.isfile(frame_file):
            count_path += 1
        else:
            cv2.imwrite(frame_file, nimg)
            print("   Save image sucessful:", frame_file)
            break

def crop_box_face(frame, detection):
    image_width, image_height = frame.shape[1], frame.shape[0]
    
    bbox = detection.location_data.relative_bounding_box
    if bbox is not None:
        bbox.xmin = bbox.xmin * image_width
        bbox.ymin = bbox.ymin * image_height
        bbox.width = bbox.width * image_width
        bbox.height = bbox.height * image_height
        
        x0 = int(bbox.xmin)
        x1 = int((bbox.xmin)+(bbox.width))
        y0 = int(bbox.ymin)
        y1 = int((bbox.ymin)+(bbox.height))
        x = abs(x1-x0)
        y = abs(y1-y0)
        y_expand = int(y/6)
        x_expand = int(x/6)
        
        x0, x1, y0, y1 = int(x0-x_expand), int(x1+x_expand), int(y0-y_expand*2), int(y1)
        cropped = frame[y0:y1, x0:x1]
        
        width, height, _ = cropped.shape
        if width != 0 and height != 0:
            ratio = width/height
            if ratio > 0.98 and ratio < 1.02:
                cropped = cv2.resize(cropped, (112, 112))
                
                # return cropped, (int(bbox.xmin), int(bbox.ymin)), (int(bbox.xmin + bbox.width), int(bbox.ymin + bbox.height))
                return cropped, (x0, y0), (x1, y1)

def stream():
    mp_facedetector = mp.solutions.face_detection
    
    fps_new_frame = 0
    fps_prev_frame = 0
    frames = 0
    
    cosine_threshold = 0.4
    proba_threshold = 0.08
    comparing_num = 5
    frame_count = 0
    
    trackers = []
    texts = []
    fake_ckeck = []
    
    # Start streaming and recording
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    # cap = cv2.VideoCapture("test.mp4")
    
    with mp_facedetector.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
        while cap.isOpened():
            _, frame = cap.read()
            frames += 1
            # Fps caculating
            fps_new_frame = time.time()
            
            # Save 5 frame if press "a" button
            if cv2.waitKey(1) == ord('a'):
                frame_count = 1
                print(">> Order received, save 10 frame to datasets/unlabel/unknow")
                _thread.start_new_thread(import_label_and_train, ())
            
            if frames%3 == 0:
                frames = 0
                trackers = []
                texts = []
                fake_ckeck = []
                
                results = face_detection.process(frame)
                if results.detections is not None:
                    for detection in results.detections:
                        result = crop_box_face(frame, detection)
                        
                        if result is not None:
                            cropped, start_bbox_point, end_bbox_point = result
                            if frame_count > 10:
                                frame_count = 0
                            if frame_count != 0:
                                _thread.start_new_thread(check_and_save_image, (cropped, ))
                                frame_count += 1
                                
                            text = ""

                            fake, score = anti_spoofing(cropped)
                            if fake:
                                text = "Fake face " + str(round(score, 2))
                                fake_ckeck.append(True)
                                texts.append(text)
                            else:
                                nimg = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
                                nimg = np.transpose(nimg, (2,0,1))
                                embedding = embedding_model.get_feature(nimg).reshape(1,-1)
                                
                                # # Predict class (SVM MODEL)
                                preds = model.predict_proba(embedding)
                                preds = preds.flatten()
                                j = np.argmax(preds)
                                proba = preds[j]
                                # Compare this vector to source class vectors to verify it is actual belong to this class
                                match_class_idx = (labels == j)
                                match_class_idx = np.where(match_class_idx)[0]
                                selected_idx = np.random.choice(match_class_idx, comparing_num)
                                compare_embeddings = embeddings[selected_idx]
                                # Calculate cosine similarity
                                cos_similarity = CosineSimilarity(embedding, compare_embeddings)
                                if cos_similarity < cosine_threshold and proba > proba_threshold:
                                    text = le.classes_[j]
                                
                            # Start tracking
                            tracker = dlib.correlation_tracker()
                            rect = dlib.rectangle(start_bbox_point[0], start_bbox_point[1], end_bbox_point[0], end_bbox_point[1])
                            tracker.start_track(frame, rect)
                            trackers.append(tracker)
                            texts.append(text)
                            fake_ckeck.append(False)
                            
                            if fake:
                                cv2.rectangle(frame, start_bbox_point, end_bbox_point, (0, 0, 255), 2)
                                cv2.putText(frame, text, (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                            else:
                                cv2.rectangle(frame, start_bbox_point, end_bbox_point, (255,255,255), 2)
                                cv2.putText(frame, text, (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
                            break
                            
            else:
                for tracker, text, fake_ in zip(trackers,texts,fake_ckeck):
                    pos = tracker.get_position()

                    # unpack the position object
                    startX = int(pos.left())
                    startY = int(pos.top())
                    endX = int(pos.right())
                    endY = int(pos.bottom())
                    
                    if fake_ is True:
                        cv2.rectangle(frame, (startX, startY), (endX, endY), (0,0,255), 1)
                        cv2.putText(frame, text, (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,255), 2)
                    else:
                        cv2.rectangle(frame, (startX, startY), (endX, endY), (255,255,255), 1)
                        cv2.putText(frame, text, (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
                        
            # Calulate fps
            fps = 1/(fps_new_frame - fps_prev_frame)
            fps_prev_frame = fps_new_frame
            cv2.putText(frame, str(round(fps, 2)), (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.imshow("Frame", frame)  
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            
            
stream()
