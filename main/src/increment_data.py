from re import I
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense
from keras.losses import categorical_crossentropy
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from tensorflow.keras.optimizers import Adam

from .get_faces_from_video import get_faces_from_video
from .faces_embedding import embedding_for_increment, embedding_all
from .train_softmax import train

import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import pickle
import argparse
import sys

sys.path.append("..")

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="outputs/my_model.h5",
                help="path to output trained model")
ap.add_argument("--le", default="outputs/le.pickle",
                help="path to output label encoder")

args = vars(ap.parse_args())

def load_label_and_embedding(embedding_path):
    # Load the face embeddings
    data = pickle.loads(open(embedding_path, "rb").read())

    # Encode the labels
    le = LabelEncoder()
    labels = le.fit_transform(data["names"])
    num_classes = len(np.unique(labels))
    labels = labels.reshape(-1, 1)
    one_hot_encoder = OneHotEncoder(categorical_features = [0])
    labels = one_hot_encoder.fit_transform(labels).toarray()
    embeddings = np.array(data["embeddings"])
    
    return labels, embeddings, num_classes
    
def edit_model_for_increment(num_label, model_path="outputs/my_model.h5"):
    # Load previous model and edit the layers
    old_model = load_model(model_path)
    model = Sequential()

    for layer in old_model.layers[:-1]:
        layer.trainable = True
        model.add(layer)

    model.add(Dense(num_label, activation='softmax', name="output_layer"))
    print(model.summary())
    return model

def train_new_model(model, labels, embeddings):
    # Initialize Softmax training model arguments
    BATCH_SIZE = 32
    EPOCHS = 20
    input_shape = embeddings.shape[1]
    
    cv = KFold(n_splits = 5, random_state = 42, shuffle=True)
    history = {'acc': [], 'val_acc': [], 'loss': [], 'val_loss': []}
    
    # model = model.build()

    optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss=categorical_crossentropy,
                    optimizer=optimizer,
                    metrics=['accuracy'])
    
    for train_idx, valid_idx in cv.split(embeddings):
        X_train, X_val, y_train, y_val = embeddings[train_idx], embeddings[valid_idx], labels[train_idx], labels[valid_idx]
        his = model.fit(X_train, y_train,
                        batch_size=BATCH_SIZE,
                        epochs=EPOCHS,
                        verbose=1,
                        validation_data=(X_val, y_val))
        try:
            print(his.history['acc'])
            history['acc'] += his.history['acc']
            history['val_acc'] += his.history['val_acc']
        except:
            print(his.history['accuracy'])
            history['acc'] += his.history['accuracy']
            history['val_acc'] += his.history['val_accuracy']
            
        history['loss'] += his.history['loss']
        history['val_loss'] += his.history['val_loss']

    # write the face recognition model to output
    model.save(args['model'])
    # f = open(args["le"], "wb")
    # f.write(pickle.dumps(le))
    # f.close()

    # Plot
    plt.figure(1)
    # Summary history for accuracy
    plt.subplot(211)
    plt.plot(history['acc'])
    plt.plot(history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    # Summary history for loss
    plt.subplot(212)
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('outputs/accuracy_loss.png')
    plt.show()

def main(video_input_path, label_name):
    # 1ST STEP: GET FACE DATA FROM VIDEO
    output_frames_folder = "datasets/train/" + str(label_name)
    # Attention: The name of output_frames_folder is using for label's name
    get_faces_from_video(video_input=video_input_path,
                        output_frames_folder=output_frames_folder)
    embedding_all()
    train()
    
'''
    # 2ND STEP: Load and edit embedding file
    embedding_path = embedding_for_increment(output_frames_folder, embedding_path="src/outputs/embeddings.pickle")

    # 3RD STEP: Get label and embedding
    labels, embeddings, num_classes = load_label_and_embedding(embedding_path)
    print("Number of classes:", num_classes)
    # 4TH STEP: Edit availabled model
    model = edit_model_for_increment(num_classes, model_path="src/outputs/my_model.h5")

    # 5TH STEP: Increment training with new data and label
    train_new_model(model, labels, embeddings)
    print(">> New model trained sucessfully!")
    
    # 6TH STEP: Test with recognize_image/stream/video.py
'''
# main("E:/Timekeeping/Face Recognition with InsightFace/datasets/videos_input/hang.mp4", "Hang")
