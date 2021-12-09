import cv2
import time
import _thread

model = "My model"

def change_model():
    global model
    time.sleep(10)
    model = "Model updated"

def main():
    # define a video capture object
    vid = cv2.VideoCapture(0)
    
    while(True):
        print(model)
        ret, frame = vid.read()
        cv2.imshow('frame', frame)
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            _thread.start_new_thread(change_model, ())
            # break
        
    vid.release()
    cv2.destroyAllWindows()
    
main()