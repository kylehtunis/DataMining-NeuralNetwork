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

print('Preprocessing\n')
###replace missing values
preprocess.missing_values(d, meta)

#use domain knowledge to group attributes
preprocess.groupByContinent(d)
preprocess.groupEducation(d)
preprocess.groupMarried(d)
#print(d)

###partition
#np.random.seed=(1)
np.random.shuffle(d)
partitions=partition.partition(d, 5)

###get training parameters

###get range of each attribute
ranges={}
for i,att in enumerate(meta.names()):
    if meta.types()[i]=='nominal':
        ranges[att]=list(set(d[att]))
    else:
        ranges[att]=['#']
#ranges['native-country']=[b'North-America', b'Central-South-America', b'Europe', b'Asia', b'Africa', b'Oceania']

###preproces and train 10 models
models=[]
for i in range(len(partitions)):
    test=partitions.pop(0)
    testCopy=test.copy()
    train=np.concatenate(partitions).copy()
    print('Training Model '+str(i+1)+' ('+str(len(train))+' samples)')
    preprocess.z_score(train, meta)
#    print(train)
    nnData=DataTransform.transform(train, meta, ranges)
#    print(nnData)
    nn = NeuralNetwork.NeuralNetwork(epochs=10, hidden=25, minError=.005, learningRate=1)
    nn.train(nnData, meta)
    models.append(nn)
    partitions.append(test)
    print()

###evaluate results
for i, model in enumerate(models):
    test=partitions[i].copy()
    preprocess.z_score(test, meta)
    nnData=DataTransform.transform(test, meta, ranges)
    #    print(nnData)
    results=model.classify(nnData, meta)
    #    print(results)
    goldLabels=DataTransform.getGoldLabels(test, meta, ranges)
    #    print(goldLabels)
    e=evaluate.Evaluator(goldLabels, results)
    e.confusionMatrices()
    e.measures()
    print('Accuracy='+str(e.getAccuracy()))
    print('Macro Precision=',e.macroPrecision)
    print('Macro Recall=',e.macroRecall)
    print('Macro F1=',e.macroF1)
    print()