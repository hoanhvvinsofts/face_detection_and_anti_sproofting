
import pickle
import dlib
import mediapipe as mp
import numpy as np
import cv2
import torch
import os
import time
import shutil
import _thread
import configparser
import datetime
import logging

from insightface.deploy import face_model
from anti_spoofing import anti_spoofing
from faces_embedding import add_embedding
from train_model import train_svm
from database_processing import database_processing

# Init logging
logging.basicConfig(filename="logging.log", level=logging.INFO, 
                    format="%(asctime)s::%(levelname)s::\t%(filename)s::%(funcName)s()::%(lineno)d\t::%(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")

console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger("logging.log").addHandler(console)
logging.info("PROGRAM STARTED!")

# Init config file
config = configparser.ConfigParser()
config.read("config.ini")

# Select torch gpu device
device_id = config["TORCH_DEVICE"]["device"]
device = torch.device(device_id if torch.cuda.is_available() else 'cpu')

# FaceModel and parameters
image_size = config["FACEMODEL"]["image_size"]
model = config["FACEMODEL"]["model"]
ga_model = config["FACEMODEL"]["ga_model"]
threshold = float(config["FACEMODEL"]["threshold"])
det = int(config["FACEMODEL"]["det"])
embedding_model = face_model.FaceModel(image_size, model, ga_model, threshold, det)
logging.info("Load Face Embedding model")

# Load saved embeddings and labels
embeddings_path = config["EMBEDDINGS_AND_LABELS"]["embeddings_path"]
labels_path = config["EMBEDDINGS_AND_LABELS"]["labels_path"]

data = pickle.loads(open(embeddings_path, "rb").read())
le = pickle.loads(open(labels_path, "rb").read())
embeddings = np.array(data["embeddings"])
labels = le.fit_transform(data["names"])

# Init mediapipe detector
mp_facedetector = mp.solutions.face_detection

# Load SVM model
svm_path = config["SVM_MODEL"]["svm_path"]
with open(svm_path, "rb") as f:
    model = pickle.load(f)

# import_label_and_train variables
temp_folder = config["SAVE_FRAME"]["temp_folder"]
train_dataset_path = config["DATASET"]["train_dataset_path"]

# crop_box_face variables
expand_box_ratio = int(config["CROP_BOX"]["expand_box_ratio"])
ratio_min = float(config["CROP_BOX"]["ratio_min"])
ratio_max = float(config["CROP_BOX"]["ratio_max"])

logging.info("Initial all necessary config and files sucessful!")

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

def import_label_and_train(unknow_folder=temp_folder):
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
    logging.info("Enter a new person: " + str(input_frame))
    target_folder = train_dataset_path + "/" + input_frame
    if os.path.isdir(target_folder):
        logging.error("This label has already exist, saved frames are not move to train folder! Try other label!")
        # print("This label has already exist, saved frames are not move to train folder! Try other label!")
    else:
        os.makedirs(target_folder)
        for img_path in os.listdir(unknow_folder):
            src = unknow_folder + "/" + img_path
            dst = target_folder + "/" + img_path
            try:
                os.rename(src, dst)
            except FileExistsError:
                logging.warning("Duplicate file. Force overwrite file:" + str(dst))
                # print("WARNING: Duplicate file. Force overwrite file:", dst)
                os.remove(dst)
                os.rename(src, dst)
                pass
            
        # print(">> New label add to Train data!")
        # print(">> Start training. . .")
        # print(">> 1 - Embedding faces. . .")
        logging.info(">> New label add to Train data!")
        logging.info(">> Start training. . .")
        logging.info(">> 1 - Embedding faces. . .")
        add_embedding(input_frame, target_folder)
        logging.info(">> 2 - Training new model. . .")
        # print(">> 2 - Training new model. . .")
        train_svm()
        logging.info(">> 3 - Reloading new model. . .")
        # print(">> 3 - Reloading new model. . .")
        with open(svm_path, "rb") as f:
            model = pickle.load(f)
            
        data = pickle.loads(open(embeddings_path, "rb").read())
        le = pickle.loads(open(labels_path, "rb").read())
        
        embeddings = np.array(data["embeddings"])
        labels = le.fit_transform(data["names"])
        logging.info(">> Loaded new model, new faces can now recognizable!")
        # print(">> Loaded new model, new faces can now recognizable!")

def check_and_save_image(nimg, folder=temp_folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    count_path = 0
    while True:
        frame_file = folder + "/unknown" + str(count_path) + ".jpg"
        if os.path.isfile(frame_file):
            count_path += 1
        else:
            cv2.imwrite(frame_file, nimg)
            # print("   Save image sucessful:", frame_file)
            logging.info("Save image sucessful: " + str(frame_file))
            break

def crop_box_face(frame, detection, expand_box_ratio=expand_box_ratio):
    '''
    `expand_box_ratio` is using for expanding the bounding box when detected.
    The bigger `expand_box_ratio` then the expand is smaller.
    default = 6
    '''
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
        y_expand = int(y/expand_box_ratio)
        x_expand = int(x/expand_box_ratio)
        
        x0, x1, y0, y1 = int(x0-x_expand), int(x1+x_expand), int(y0-y_expand*2), int(y1)
        cropped = frame[y0:y1, x0:x1]
        
        width, height, _ = cropped.shape
        if width != 0 and height != 0:
            ratio = width/height
            if ratio > ratio_min and ratio < ratio_max:
                cropped = cv2.resize(cropped, (112, 112))
                return cropped, (x0, y0), (x1, y1)

def stream():
    fps_new_frame = 0
    fps_prev_frame = 0
    frames = 0
    frame_count = 0
    trackers = []
    texts = []
    fake_ckeck = []
    
    cosine_threshold = float(config["THRESHOLD_PREDICT"]["cosine_threshold"])
    proba_threshold = float(config["THRESHOLD_PREDICT"]["proba_threshold"])
    comparing_num = int(config["THRESHOLD_PREDICT"]["comparing_num"])

    frame_width = int(config["RESOLUTION_FRAME"]["frame_width"])
    frame_height = int(config["RESOLUTION_FRAME"]["frame_height"])
    
    model_selection = int(config["FACE_DETECTION_MODEL"]["model_selection"])
    min_detection_confidence = float(config["FACE_DETECTION_MODEL"]["min_detection_confidence"])
    
    quit_btn = config["BUTTONS"]["quit"]
    save_frame_btn = config["BUTTONS"]["save_frame"]
    save_frame_number = int(config["SAVE_FRAME"]["save_frame_number"])
    processing_frame_every = int(config["PROCESSING_FRAME_EVERY"]["processing_frame_every"])
    
    # Start streaming and recording
    # cap = cv2.VideoCapture(0)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
    cap = cv2.VideoCapture("hoan.mp4")
    
    with mp_facedetector.FaceDetection(model_selection=model_selection, min_detection_confidence=min_detection_confidence) as face_detection:
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                frames += 1
                # Fps caculating
                fps_new_frame = time.time()
                
                # Save frames if press save_frame_btn button
                if cv2.waitKey(1) == ord(save_frame_btn):
                    frame_count = 1
                    logging.info(">> Order received, save" + str(save_frame_number) + " frame to [TEMP_FOLDER]")
                    # print(">> Order received, save" + str(save_frame_number) + " frame to [TEMP_FOLDER]")
                    _thread.start_new_thread(import_label_and_train, ())

                if frames%processing_frame_every == 0:
                    frames = 0
                    trackers = []
                    texts = []
                    fake_ckeck = []
                    
                    results = face_detection.process(frame)
                    if results.detections is not None:
                        for detection in results.detections:
                            result = crop_box_face(frame, detection, expand_box_ratio)
                            if result is not None:
                                cropped, start_bbox_point, end_bbox_point = result
                                if frame_count > save_frame_number:
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
                                        recognize_time = datetime.datetime.now().strftime('%d-%m-%Y %H:%M:%S')
                                        database_processing(text, recognize_time)
                                        
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
                if cv2.waitKey(1) & 0xFF == ord(quit_btn):
                    break
            else:
                logging.warning("Camera or video is disabled")
                break
            
stream()
