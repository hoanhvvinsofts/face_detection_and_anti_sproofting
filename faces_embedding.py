import sys
sys.path.append('insightface/deploy')
sys.path.append('insightface/src/common')

from imutils import paths
import numpy as np
import face_model
import pickle
import cv2
import os

import configparser

config = configparser.ConfigParser()
config.read("config.ini")

# FaceModel and parameters
image_size = config["FACEMODEL"]["image_size"]
model = config["FACEMODEL"]["model"]
ga_model = config["FACEMODEL"]["ga_model"]
threshold = float(config["FACEMODEL"]["threshold"])
det = int(config["FACEMODEL"]["det"])
embedding_model = face_model.FaceModel(image_size, model, ga_model, threshold, det)

train_dataset_path = config["DATASET"]["train_dataset_path"]

# Load saved embeddings and labels
embeddings_path = config["EMBEDDINGS_AND_LABELS"]["embeddings_path"]
labels_path = config["EMBEDDINGS_AND_LABELS"]["labels_path"]

def faces_embedding():
    # Grab the paths to the input images in our dataset
    imagePaths = list(paths.list_images(train_dataset_path))

    # Initialize our lists of extracted facial embeddings and corresponding people names
    knownEmbeddings = []
    knownNames = []

    # Loop over the imagePaths
    for (i, imagePath) in enumerate(imagePaths):
        # extract the person name from the image path
        name = imagePath.split(os.path.sep)[-2]

        # load the image
        image = cv2.imread(imagePath)
        # convert face to RGB color
        nimg = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        nimg = np.transpose(nimg, (2,0,1))
        # Get the face embedding vector
        face_embedding = embedding_model.get_feature(nimg)
        
        # add the name of the person + corresponding face
        # embedding to their respective list
        knownNames.append(name)
        knownEmbeddings.append(face_embedding)

    # save to output
    data = {"embeddings": knownEmbeddings, "names": knownNames}
    f = open(embeddings_path, "wb")
    f.write(pickle.dumps(data))
    f.close()

def add_embedding(new_label_name, target_data_folder):
    # Load the face embeddings
    data = pickle.loads(open(embeddings_path, "rb").read())
    knownNames = data["names"]     # List of labels name
    knownEmbeddings = data["embeddings"] # List of embeddings
    
    for img_path in os.listdir(target_data_folder):
        img_path = target_data_folder + "/" + img_path
        
        # load the image
        image = cv2.imread(img_path)
        # convert face to RGB color
        nimg = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        nimg = np.transpose(nimg, (2,0,1))
        # Get the face embedding vector
        face_embedding = embedding_model.get_feature(nimg)
        
        knownNames.append(new_label_name)
        knownEmbeddings.append(face_embedding)
        
    # save to output
    data = {"embeddings": knownEmbeddings, "names": knownNames}
    f = open(embeddings_path, "wb")
    f.write(pickle.dumps(data))
    f.close()
