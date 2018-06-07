
from keras.models import Sequential
from keras.layers import Flatten,Dense, Dropout,Conv2D, Activation, MaxPooling2D, Convolution1D, MaxPooling1D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam,SGD
import keras.backend as K
import matplotlib.image as mpimg
import numpy as np
import scipy.io
import tensorflow as tf 
import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from keras.optimizers import Adam,SGD
def make_model(load=False, filepath='./Saved_model', optim=None):
    #Preventing none parameter
    if(optim == None):
        optim = 'adam'
    model = Sequential()
    model.add(Convolution1D(
        nb_filter=32, 
        filter_length=2, 
        input_dim=1,  # Should this be 1 or 252?
        input_length=15,
        activation='relu',
        init='uniform'))
    model.add(Convolution1D(
        nb_filter=64, 
        filter_length=2, 
        input_dim=1,  # Should this be 1 or 252?
        input_length=32,
        activation='relu',
        init='uniform'))
    model.add(Convolution1D(
        nb_filter=124, 
        filter_length=2, 
        input_dim=1,  # Should this be 1 or 252?
        input_length=64,
        activation='relu',
        init='uniform'))
    model.add(Convolution1D(
        nb_filter=32, 
        filter_length=2, 
        input_dim=1,  # Should this be 1 or 252?
        input_length=124,
        activation='relu',
        init='uniform'))


    model.add(Flatten())
    model.add(Dense(units=512,  activation='softmax'))
    model.add(Dense(units=15,  activation='softmax'))

    if(load):
        model.load_weights(filepath)

    model.compile(optimizer=optim,
                  loss=custom_loss)
    return model

def custom_loss(y_true, y_pred): 
    error = K.mean(K.abs(y_pred - y_true ))
    return error


def training(Discharge_Summary):
    common_diseases = np.load("common_diseases.npy")
    # Extract all sentences from the Discharge summaries
    rgx = re.compile("([\w][\w']*\w)")
    strings = list()
    for i in Discharge_Summary:

        if(str(i) == "nan"):
            continue
        strings+= re.split("\.\s*",i.lower())
    all_sentences = list(strings)
    for ind in range(len(all_sentences)):

        all_sentences[ind] = rgx.findall(all_sentences[ind])
        
    sentences_with_disease = list()
    new_sentence = None
    for j,sentence in enumerate(all_sentences):
        for disease in common_diseases:
            if(np.where(np.array(sentence) == disease)[0].size):
    #             location = np.where(np.array(sentence) == disease)[0][0]
                new_sentence = sentence
                sentences_with_disease += [new_sentence]
                ind+=1
                break
    training_sentences = list()
    for sentence in sentences_with_disease:
        while(len(sentence) != 0):
            if(len(sentence) < 15):
                sentence+=list(np.zeros(15-len(sentence)))
            else:
                training_sentences += [sentence[:15]]
                sentence = sentence[15:]
    labels = np.zeros((len(training_sentences),15))
    for ind, sentence in enumerate(training_sentences):
        for disease in common_diseases:
            if(np.where(np.array(sentence) == disease)[0].size):
                locations = np.where(np.array(sentence) == disease)
                for ind2 in locations:
                    labels[ind][ind2] = 1
                break
    return training_sentences,labels
def num_p(training_sentences,dic):
    num_representation = list()
    for sentence in training_sentences:
        temp_list = list()
        for word in sentence:
            if word in dic:
                temp_list += [dic[str(word)]]
            else:
                temp_list += [0]
        num_representation+=[temp_list]
    return num_representation
      
    

    
def predict(last=0,discharge=0):
    print(1)
    
    print(2)
    
    if(last!=0):
        pandaV = pd.read_csv("NOTEEVENTS.csv",low_memory=False)
        NoteEvents_Data= np.array(pandaV)
        Discharge_Summary = NoteEvents_Data[NoteEvents_Data[:,6] == "Discharge summary",10][last-1:last]
    if(discharge!=0):
        Discharge_Summary=discharge
    print(3)
    words = np.load("words.npy")
    print(4)
    dic = {word: i+1 for i,word in enumerate(words)}
    dic['0.0']=0
    print(5)
    training_sentences,labels = training(Discharge_Summary)
    print(6)
    num_representation = num_p(training_sentences,dic)
    print(7)
    adam = Adam(lr=.0001, decay=.0)
    model = make_model(load=True,filepath='model_weights.h10',optim=adam)
    print(8)
    a = model.predict(np.array(num_representation).reshape(np.array(num_representation).shape + (1,)), batch_size=100)
    b = training_sentences
    print(10)
    
    diseases = list()
    for i,sentence in enumerate(a):
        if 1 in list(sentence>.999):
            ind = list(sentence>.999).index(1)
            diseases+=[b[i][ind]]
    return a, b, diseases