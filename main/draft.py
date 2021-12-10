import cv2
import time
import _thread
from tensorflow.keras.models import load_model
from faces_embedding import faces_embedding
from train_softmax import train_softmax

model = load_model("src/outputs/my_model.h5")

def change_model():
    global model
    print(model)
    print(">> 1 - Embedding faces. . .")
    faces_embedding()
    print(">> 2 - Training new model. . .")
    train_softmax()
    print(">> 3 - Reloading new model. . .")
    model = load_model("src/outputs/my_model.h5")
    print(model)
    exit()

def main():
    # define a video capture object
    vid = cv2.VideoCapture(0)
    
    while(True):
        print(model)
        ret, frame = vid.read()
        cv2.imshow('frame', frame)
        
        if cv2.waitKey(1) & 0xFF == ord("a"):
            _thread.start_new_thread(change_model, ())
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        
    # vid.release()
    # cv2.destroyAllWindows()
    
main()