import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import csv
import ast
import os
from pandas import Series,DataFrame
from collections import defaultdict

from imblearn.datasets import make_imbalance
from imblearn.over_sampling import SMOTE 
from imblearn.under_sampling import CondensedNearestNeighbour 
#-------my file------
from embedding import *
from util import get_code_from_file,split_hold_txt,print_results,evaluate_model
from dl import *
from test import *

def building_data_map(testfile,csvfile = 'linuxcommitfilechange.csv'):
    '''
        This function will read the map csv file, and training data to map whether code is vulnerability tag
        1 means yes otherwhile no
        ------------------------------------------
        args:
            testfile : .txt file ( code(int), tag(1/0)) 
            csvfile : .csv file (code filename functionName)
        return:
            dictionary: code, file name, function name, vun target   
    '''
    # open csv file and saving data: 
    d = defaultdict(list)
    temp = defaultdict(list)
    for line in open(testfile, "r"):
        a = line.replace('\n', '').split('\t')
        #list[262677   0]
        temp[a[0]] = a[1]

    with open(csvfile, mode='r') as csvfile_reader:
        for row in csv.reader(csvfile_reader):
            csv_data = row[0].split('\t')
            if(csv_data[0] in temp.keys()):
                d[csv_data[0]].append(csv_data[1])
                d[csv_data[0]].append(csv_data[2])
                d[csv_data[0]].append(temp[csv_data[0]])                
    return d

def file_to_map_and_DF(filepath):
    '''
        According to map we have, iterate the code to dataframe from 
        all file
    '''
    fileDic = building_data_map(filepath)
    keyList,codeList,targetList = ([] for i in range(3))
    for key in fileDic.keys():
        keyList.append(key)
        codeList.append(get_code_from_file(fileDic, key))
        targetList.append(fileDic[key][2])
    data = {
        "key":keyList,
        "code":codeList,
        "target":targetList
    }
    return fileDic,DataFrame(data)

def make_x_and_y(df,w2v_filename):
    w2v = KeyedVectors.load(w2v_filename, mmap='r') 
    #Build X and Y
    x = np.random.rand(len(df),100)
    #iterate all dataframe, let each x has vector that insert
    for i in range(len(df)):
        k = 0 
        non = 0
        values = np.zeros(100) # Each vector
        for word in tfidf(df['code'].iloc[i]):#all tfidf hightest value top key = 3
            #print(tfidf(getattr(row, 'code')))
            if word in w2v:
                values += w2v[word]
                k+=1
        if k>0:
            x[i,:]=values/k
        else: 
            non+=1
    y = LabelEncoder().fit_transform(df['target'].values)
    return x,y


from imblearn.combine import SMOTETomek 

from imblearn.under_sampling import EditedNearestNeighbours 
from imblearn.under_sampling import RandomUnderSampler 

if __name__ == "__main__":
    '''
    Aim:
        - what's the performance of different ML algorithms on security vulnerability prediction?
        - what are the impact of the parameters of ML algorithms?
        - what's the performance of different data imbalance techniques on security vulnerability prediction?
        冲鸭，你是最棒的 -April 5 4pm 2022
    '''
    
    # #Data collection
    trainingfile = './training-test-pairs/0/training_patches' #Training file
    testfile = './training-test-pairs/0/test_patches' #Test file
    
    # #Building dictionary and dataframe
    trainFD,trainDF = file_to_map_and_DF(trainingfile)
    testFD, testDF  = file_to_map_and_DF(testfile)
    # if not os.path.exists("./w2v.wordVector"):
    #     #Generate code feature
    
    training_embedding(trainFD,trainDF,'w2v.wordVector',size = 100, window=10,sg = 1,min_count = 8)
    training_embedding(trainFD,trainDF,'w2v.wordVector2',size = 100, window=10,sg = 1,min_count = 8)

    
    #X: the set of vector that exist speical feature   |  Y: 1 / 0
    X_train,y_train= make_x_and_y(trainDF, 'w2v.wordVector') 
    X_test ,y_test = make_x_and_y(testDF, 'w2v.wordVector2')

    #imbalance
    X_res = X_train
    y_res = y_train

    # sm = SMOTE(random_state=42)
    # X_res, y_res = sm.fit_resample(X_train, y_train)

    cnn = CondensedNearestNeighbour(random_state=42)
    X_res, y_res = cnn.fit_resample(X_train, y_train)

    # rus = RandomUnderSampler(random_state=42)
    # X_res, y_res = rus.fit_resample(X_train, y_train)
    # smt = SMOTETomek(random_state=42)
    # X_res, y_res = smt.fit_resample(X_train, y_train)
    # randomF(X_res, y_res,X_test ,y_test)
    # lr(X_res, y_res,X_test ,y_test)
    # navbee(X_res, y_res,X_test ,y_test)
    svm(X_res, y_res,X_test ,y_test)
    mlp(X_res, y_res,X_test ,y_test)
    # LSTM(X_res, y_res, X_test, y_test)



    #将每个training的代码token和语义库比较，如果有，就输出他的vector

    # for i in range(len(x)):
    #     k = 0
    #     non = 0
    #     values = np.zeros(test_df.shape[0])
    #     for code in split_hold_txt(test_df['code'].iloc[i],split_sen=False):
    #         print(code)

    #Train the model


    
    #Eliminate data imbalance(try one first and other can be done after ML model training finish)

    #1. Under-sampling

    #2. Over-sampling

    #3. Over-sampling followed by under-sampling

    #4. Ensemble classifier using samplers internally


    # X, y = make_imbalance(
    #     data.drop("target", axis='columns'),
    #     data.target,
    #     sampling_strategy={0: 50, 1: 50},
    #     random_state=RANDOM_STATE,
    # )
    #X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=RANDOM_STATE)
   
    
    #training in ML


    #Testing
