from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import numpy as np
import pickle
import configparser

config = configparser.ConfigParser()
config.read("config.ini")

# Load saved embeddings and labels
embeddings_path = config["EMBEDDINGS_AND_LABELS"]["embeddings_path"]
labels_path = config["EMBEDDINGS_AND_LABELS"]["labels_path"]

# SVM model varriables
kernel = config["SVM_MODEL"]["kernel"]
probability = config["SVM_MODEL"].getboolean("probability")
max_iter = int(config["SVM_MODEL"]["max_iter"])
svm_path = config["SVM_MODEL"]["svm_path"]

def train_svm():
    # Load the face embeddings
    data = pickle.loads(open(embeddings_path, "rb").read())

    le = LabelEncoder()
    labels = le.fit_transform(data["names"])
    embeddings = np.array(data["embeddings"])

    # Build SVC Classifier
    model = SVC(kernel=kernel, probability=probability, max_iter=max_iter)
    model.fit(embeddings, labels)

    # write the face recognition model to output
    with open(svm_path,'wb') as f:
        pickle.dump(model, f)
        
    with open(labels_path,'wb') as f:
        pickle.dump(le, f)
