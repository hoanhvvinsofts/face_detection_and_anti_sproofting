# '''
from facenet_pytorch import MTCNN

import cv2
import torch
import time

# device =  torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# print(device)

mtcnn = MTCNN(thresholds= [0.7, 0.7, 0.8] ,keep_all=True, device = 'cuda')

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)

while cap.isOpened():
    isSuccess, frame = cap.read()
    if isSuccess:
        bbox_start = time.time()
        boxes, _ = mtcnn.detect(frame)
        bbox_end = time.time()
        print("Bbox time cost:", bbox_end-bbox_start)
        if boxes is not None:
            for box in boxes:
                bbox = list(map(int,box.tolist()))
                frame = cv2.rectangle(frame,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(0,0,255),6)
    cv2.imshow('Face Detection', frame)
    if cv2.waitKey(1)&0xFF == 27:
        break
    
cap.release()
cv2.destroyAllWindows()
# '''

# import torch

# print(torch.cuda.is_available())

# print(torch.cuda.current_device())