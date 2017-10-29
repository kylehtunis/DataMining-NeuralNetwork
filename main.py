# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 18:21:04 2017

@author: kyleh
"""

import numpy as np
import scipy.io.arff as arff
import sys
import partition
import NeuralNet
import DataTransform
import evaluate

###get data from file
fileName=sys.argv[1]
f=open(fileName, 'r')
arff.loadarff(f)

###partition
d = data.copy()
numpy.random.seed=(80085)
numpy.random.shuffle(d)
partitions=partition.partition(d)

###preproces and train 10 models
models=[]
for i in range(len(partitions)):
    print('Training Fold '+str(i+1))
    test=partitions.pop(0)
    testCopy=test.copy()
    train=np.concatenate(partitions).copy()
    preprocess.preprocess(train, meta)
    nnData=DataTransform.transform(train, meta)
    NeuralNet nn(nnData)
    models.append(nn)
    partitions.append(testCopy)
    
###classify test data
results[]
for i in range(len(models)):
    results.append(m.classify(partitions[i]))
    
###evaluate results
for i in range(len(results)):
    evaluate.evaluate(results[i], partitions[i])