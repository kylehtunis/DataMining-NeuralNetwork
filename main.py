# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 18:21:04 2017

@author: kyleh
"""

import numpy as np
import scipy.io.arff as arff
import sys
import partition
import NeuralNetwork
import DataTransform
import evaluate
import preprocess

###get data from file
fileName=sys.argv[1]
f=open(fileName, 'r')
data, meta = arff.loadarff(f)
d = data.copy()

###replace missing values
preprocess.missing_values(d, meta)

###partition
#np.random.seed=(8008)
np.random.shuffle(d)
partitions=partition.partition(d, 5)

###get training parameters

###preproces and train 10 models
models=[]
for i in range(len(partitions)):
    test=partitions.pop(0)
    testCopy=test.copy()
    train=np.concatenate(partitions).copy()
    print('Training Model '+str(i+1)+' ('+str(len(train))+' samples)')
    preprocess.z_score(train, meta)
#    print(train)
    nnData=DataTransform.transform(train, meta)
    nn = NeuralNetwork.NeuralNetwork(epochs=1, hidden=5)
    nn.train(nnData, meta)
    models.append(nn)
    partitions.append(testCopy)
    print()
    
###classify test data
results=[]
goldLabels=[]
for i in range(len(models)):
    results.append([np.argmax(models[i].classify(partitions[i], meta)[j]) for j in range(len(partitions[i]))])
    goldLabels.append([np.argmax(DataTransform.getGoldLabels(partitions[i], meta)[j]) for j in range(len(partitions[i]))])
#    for j in range(len(partitions[i])):
#        print(goldLabels[j])
#        print(results[i][j])
    ###evaluate results
    e=evaluate.Evaluator(goldLabels[i], results[i])
    print('Accuracy='+str(e.getAccuracy()))