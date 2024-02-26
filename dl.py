import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import itertools as it

from util import *

from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt
from nltk.tokenize import WordPunctTokenizer
import tensorflow_datasets as tfds
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense
from tensorflow import keras
from tensorflow.keras import models, layers, losses, optimizers, Sequential
from sklearn.preprocessing import OneHotEncoder
#from tfa.metrics import F1Score
def randomF(X_train,y_train,X_test,y_test):
    clf = RandomForestClassifier(max_depth=2, random_state=0)
    clf.fit(X_train, y_train)
    evaluate_model("Random forest",clf,X_test,y_test)

def navbee(X_train,y_train,X_test,y_test):
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    evaluate_model("nb",gnb,X_test,y_test)

def lr(X_train,y_train,X_test,y_test):
    lr = LogisticRegression(C = 0.001, max_iter=10000,random_state=5
    ).fit(X_train, y_train)
    evaluate_model("lr",lr,X_test,y_test)


def mlp(X_train,y_train,X_test,y_test):
    print("MLP Excute")
    mlpParameter(X_train,y_train)
    mlp = MLPClassifier(hidden_layer_sizes=(100,),activation='tanh',learning_rate='constant')
    mlp.fit(X_train, y_train)
    evaluate_model("MLP",mlp,X_test,y_test)

def mlpParameter(x,y):
    #,hidden_layer_sizes=(50,),activation='relu',learning_rate='constant'
    mlp = MLPClassifier(max_iter=10000)
    params = {
        'hidden_layer_sizes': [(10,), (50,), (100,)],
        'activation': ['relu', 'tanh', 'logistic'],
        'learning_rate': ['constant', 'invscaling', 'adaptive']
    }
    cv = GridSearchCV(mlp, params, cv=5)
    cv.fit(x, y)
    print_results(cv)

def svm(X_train,y_train,X_test,y_test):
    svm = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    svm.fit(X_train, y_train)
    evaluate_model("SVM",svm,X_test,y_test)


def LSTM(x_train,y_train,x_test,y_test):

    model = Sequential([
        layers.Embedding(128, 64),
        layers.Bidirectional(layers.LSTM(64)),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])

    model.compile(loss=losses.BinaryCrossentropy(from_logits=True),
              optimizer=optimizers.Adam(1e-4),
              metrics=['acc'])

    history = model.fit(x_train,y_train, epochs=10,
                    validation_data=(x_test, y_test), 
                    validation_steps=30)
    plt.rcParams['font.size'] = 14

    #evaluate_model("NSML",model,x_test,y_test)

    plot_graphs(history, 'acc')
    plot_graphs(history, 'loss')

def plot_graphs(history, metric):
    plt.plot(history.history[metric])
    plt.plot(history.history['val_'+metric], '')
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend([metric, 'val_'+metric])
    plt.show()