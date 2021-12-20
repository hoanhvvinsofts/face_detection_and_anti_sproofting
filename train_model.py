from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import numpy as np
import pickle

def train_svm():
    # Load the face embeddings
    data = pickle.loads(open("src/outputs/embeddings.pickle", "rb").read())

    le = LabelEncoder()
    labels = le.fit_transform(data["names"])
    embeddings = np.array(data["embeddings"])

    # Build SVC Classifier
    model = SVC(kernel='linear', probability=True, max_iter=-1)
    model.fit(embeddings, labels)

    # write the face recognition model to output
    with open("src/outputs/model.pkl",'wb') as f:
        pickle.dump(model, f)
        
    with open("src/outputs/le.pickle",'wb') as f:
        pickle.dump(le, f)
