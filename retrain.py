from faces_embedding import faces_embedding
from train_model import train_softmax, train_svm

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

faces_embedding()
train_svm()