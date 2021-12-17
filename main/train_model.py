from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from src.softmax import SoftMax
from sklearn.svm import SVC
import numpy as np
import pickle

def train_softmax():
    # Load the face embeddings
    data = pickle.loads(open("src/outputs/embeddings.pickle", "rb").read())

    le = LabelEncoder()
    labels = le.fit_transform(data["names"])
    num_classes = len(np.unique(labels))
    labels = labels.reshape(-1, 1)
    one_hot_encoder = OneHotEncoder(categorical_features = [0])
    labels = one_hot_encoder.fit_transform(labels).toarray()

    embeddings = np.array(data["embeddings"])

    # Initialize Softmax training model arguments
    BATCH_SIZE = 32
    EPOCHS = 20
    input_shape = embeddings.shape[1]

    # Build sofmax classifier
    softmax = SoftMax(input_shape=(input_shape,), num_classes=num_classes)
    model = softmax.build()
    
    # Create KFold
    cv = KFold(n_splits = 5, random_state = 42, shuffle=True)
    history = {'acc': [], 'val_acc': [], 'loss': [], 'val_loss': []}
    # Train
    for train_idx, valid_idx in cv.split(embeddings):
        X_train, X_val, y_train, y_val = embeddings[train_idx], embeddings[valid_idx], labels[train_idx], labels[valid_idx]
        model.fit(X_train, y_train,
                        batch_size=BATCH_SIZE,
                        epochs=EPOCHS,
                        verbose=0,
                        validation_data=(X_val, y_val))
  
    # write the face recognition model to output
    model.save("src/outputs/my_model.h5")
    f = open("src/outputs/le.pickle", "wb")
    f.write(pickle.dumps(le))
    f.close()

def train_svm():
    # Load the face embeddings
    data = pickle.loads(open("src/outputs/embeddings.pickle", "rb").read())

    le = LabelEncoder()
    labels = le.fit_transform(data["names"])
    embeddings = np.array(data["embeddings"])

    # Build SVC Classifier
    model = SVC(kernel='linear', probability=True)
    model.fit(embeddings, labels)

    # write the face recognition model to output
    with open("src/outputs/model.pkl",'wb') as f:
        pickle.dump(model, f)
        
    with open("src/outputs/le.pickle",'wb') as f:
        pickle.dump(le, f)
