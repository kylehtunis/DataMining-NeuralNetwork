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
preprocess.preprocess(d, meta)

###partition
np.random.seed=(80085)
np.random.shuffle(d)
partitions=partition.partition(d)

###preproces and train 10 models
models=[]
for i in range(len(partitions)):
    print('Training Fold '+str(i+1))
    test=partitions.pop(0)
    testCopy=test.copy()
    train=np.concatenate(partitions).copy()
    preprocess.z_score(train, meta)
#    print(train)
    nnData=DataTransform.transform(train, meta)
    nn = NeuralNetwork.NeuralNetwork()
    models.append(nn)
    partitions.append(testCopy)
    print()
    
###classify test data
results=[]
for i in range(len(models)):
    results.append(models[i].classify(partitions[i]))
    
###evaluate results
for i in range(len(results)):
    evaluate.evaluate(results[i], partitions[i])