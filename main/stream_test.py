import sys
sys.path.append('insightface/deploy')
sys.path.append('insightface/src/common')

from facenet_pytorch import MTCNN
# from mtcnn.mtcnn import MTCNN
from tensorflow.keras.models import load_model
from anti_sproofing import anti_sproofing
from imutils import paths
import face_preprocess
import numpy as np
import face_model
import argparse
import pickle
import time
import dlib
import cv2
import os
import _thread



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


# Load embeddings and labels
data = pickle.loads(open(args.embeddings, "rb").read())
le = pickle.loads(open(args.le, "rb").read())

embeddings = np.array(data['embeddings'])
labels = le.fit_transform(data['names'])

# Initialize detector
# detector = MTCNN()
detector = MTCNN(thresholds= [0.7, 0.7, 0.8] ,keep_all=True, device = 'cuda')



# Initialize faces embedding model
embedding_model = face_model.FaceModel(args)

# Load the classifier model
model = load_model(args.mymodel)

# Define distance function
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

def check_and_save_image(frame, folder="datasets/unlabel/unknown"):
    if not os.path.exists(folder):
        os.makedirs(folder)
    count_path = 0
    while True:
        frame_file = folder + f"/unknown{count_path}.jpg"
        if os.path.isfile(frame_file):
            count_path += 1
        else:
            cv2.imwrite(frame_file, frame)
            break

def import_label_and_move_to_train_dir(unknow_folder="datasets/unlabel/unknown"):
    input_frame = input(">> Input label name: ")
    time.sleep(5)
    target_folder = "datasets/train/" + input_frame
    if os.path.isdir(target_folder):
        print("This label has already exist, try other label.")
    else:
        os.makedirs(target_folder)
        for img_path in os.listdir(unknow_folder):
            src = unknow_folder + "/" + img_path
            dst = target_folder + "/" + img_path
            try:
                os.rename(src, dst)
            except FileExistsError:
                print("WARNING: Duplicate file inside folder: ", target_folder)
                pass

def stream():
    # Initialize some useful arguments
    cosine_threshold = 0.8
    proba_threshold = 0.85
    comparing_num = 5
    trackers = []
    texts = []
    frames = 0
    
    prev_frame_time = 0
    new_frame_time = 0
    
    # Start streaming and recording
    cap = cv2.VideoCapture(0)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    save_width = 600
    save_height = int(600/frame_width*frame_height)
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        frames += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (save_width, save_height))
        
        new_frame_time = time.time()
        
        # Save 5 frame if press "a" button
        if cv2.waitKey(33) == ord('a'):
            frame_count = frame_count + 1
            print("Order received, save 5 frame to datasets/unlabel/unknow")
            _thread.start_new_thread(import_label_and_move_to_train_dir, ())
            
        if frame_count >= 10 and frame_count != 0:
            frame_count = 0
        elif frame_count != 0:
            if frames%3 == 0:
                _thread.start_new_thread(check_and_save_image, (frame, ))
                frame_count += 1
        
        if frames%3 == 0:
            trackers = []
            texts = []
            
            # bbox_start = time.time()                    #TIME
            # bboxes = detector.detect_faces(frame)
            bboxes, _, landmarks = detector.detect(frame, landmarks=True)
            # bbox_end = time.time()                      #TIME
            # print("Bbox time cost:", bbox_end-bbox_start)
            
            # '''
            # if len(bboxes) != 0:
            if bboxes is not None:
                
                # start = time.time()
                
                # for bboxe in bboxes:
                for bboxe, landmark in zip(bboxes, landmarks):
                    # embedding_start = time.time()                   #TIME
                    bbox = list(map(int,bboxe.tolist()))
                    
                    bbox = np.array([bbox[0], bbox[1], bbox[2], bbox[3]])
                    landmarks = landmark
                    landmarks = np.array([landmarks[0][0], landmarks[1][0], landmarks[2][0], landmarks[3][0], landmarks[4][0],
                                        landmarks[0][1], landmarks[1][1], landmarks[2][1], landmarks[3][1], landmarks[4][1]])
                    
                    landmarks = landmarks.reshape((2,5)).T
                    nimg = face_preprocess.preprocess(frame, bbox, landmarks, image_size='112,112')
                    nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
                    nimg = np.transpose(nimg, (2,0,1))
                    embedding = embedding_model.get_feature(nimg).reshape(1,-1)
                    # embedding_end = time.time()                   #TIME
                    # print("Embedding time cost:", embedding_end-embedding_start)
                    text = ""
                    
                    # anti_sproofing_start = time.time()            #TIME
                    fake, score = anti_sproofing(frame)
                    # anti_sproofing_end = time.time()              #TIME
                    # print("Anti sproffing time cost:", anti_sproofing_end-anti_sproofing_start)
                    
                    if fake:
                        text = "Fake face " + str(score)
                        
                    else:
                        # Predict class
                        
                        # predict_start = time.time()                 #TIME
                        preds = model.predict(embedding)
                        preds = preds.flatten()
                        # Get the highest accuracy embedded vector
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
                            name = le.classes_[j]
                            text = "{}".format(name)
                        
                        # predict_end = time.time()                   # TIME
                        # print("Predict time cost:", predict_end-predict_start)
                    
                    # Start tracking
                    tracker = dlib.correlation_tracker()
                    rect = dlib.rectangle(bbox[0], bbox[1], bbox[2], bbox[3])
                    tracker.start_track(rgb, rect)
                    trackers.append(tracker)
                    texts.append(text)

                    y = bbox[1] - 10 if bbox[1] - 10 > 10 else bbox[1] + 10
                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255,255,255), 2)
                    if fake:
                        cv2.putText(frame, text, (bbox[0], y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                    else:
                        cv2.putText(frame, text, (bbox[0], y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 2)
                    # '''
                    
                # end = time.time()
                # print("Time cost:", end-start)
                
        else:
            for tracker, text in zip(trackers,texts):
                pos = tracker.get_position()

                # unpack the position object
                startX = int(pos.left())
                startY = int(pos.top())
                endX = int(pos.right())
                endY = int(pos.bottom())

                cv2.rectangle(frame, (startX, startY), (endX, endY), (255,255,255), 1)
                cv2.putText(frame, text, (startX, startY - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 1)
        
        fps = 1/(new_frame_time-prev_frame_time)
        prev_frame_time = new_frame_time
        cv2.putText(frame, str(fps), (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.imshow("Frame", frame)
        # video_out.write(frame)
        # print("Faces detection time: {}s".format(detect_tock-detect_tick))
        # print("Faces recognition time: {}s".format(reco_tock-reco_tick))
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        
    # video_out.release()
    cap.release()
    cv2.destroyAllWindows()
stream()

