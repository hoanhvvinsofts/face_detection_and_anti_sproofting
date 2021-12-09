import sys
import argparse
sys.path.append('insightface/deploy')
sys.path.append('insightface/src/common')

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="datasets/train",
                help="Path to training dataset")
ap.add_argument("--embeddings", default="outputs/embeddings.pickle")
# Argument of insightface
ap.add_argument('--image-size', default='112,112', help='')
ap.add_argument('--model', default='insightface/models/model-y1-test2/model,0', help='path to load model.')
ap.add_argument('--ga-model', default='', help='path to load model.')
ap.add_argument('--gpu', default=0, type=int, help='gpu id')
ap.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from begining')
ap.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
ap.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')
args = ap.parse_args()

import face_model
from kerassurgeon.operations import replace_layer
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from imutils import paths
import numpy as np
import pandas as pd
import pickle
import os
import cv2

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

def train():
    # Load embedding and return values
    embedding_path = "src/outputs/embeddings.pickle"
    labels, embeddings, num_classes = load_label_and_embedding(embedding_path)

    # Load model and change layers
    mymodel = "src/outputs/my_model.h5"
    model = load_model(mymodel)

    weight = model.get_weights()
    output_layer = Dense(num_classes, activation='softmax', name="output_layer")
    model = replace_layer(model, model.layers[-1], output_layer)

    # Retrain/Continue training with new data
    BATCH_SIZE = 32
    EPOCHS = 20
    history = {'acc': [], 'val_acc': [], 'loss': [], 'val_loss': []}

    optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss=categorical_crossentropy,
                    optimizer=optimizer,
                    metrics=['accuracy'])

    cv = KFold(n_splits = 5, random_state = 42, shuffle=True)
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

def embedding_new_face(output_frames_folder, embedding_model, embeddings_picke_path=r"embeddings.pickle"):
    label = os.path.split(output_frames_folder)[-1]
    imagePaths = list(paths.list_images(output_frames_folder))
    embedding = pd.read_pickle(embeddings_picke_path)
    
    # Loop over the imagePaths
    for (i, imagePath) in enumerate(imagePaths):

        # load the image
        image = cv2.imread(imagePath)
        # convert face to RGB color
        nimg = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        nimg = np.transpose(nimg, (2,0,1))
        # Get the face embedding vector
        face_embedding = embedding_model.get_feature(nimg)
        
        embedding["names"].append(label)
        embedding["embeddings"].append(face_embedding)
    f = open(embeddings_picke_path, "wb")
    f.write(pickle.dumps(embedding))
    return embedding

embeddings_picke_path=r"embeddings.pickle"
embedding = pd.read_pickle(embeddings_picke_path)
print(embedding["names"])