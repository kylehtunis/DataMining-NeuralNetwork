# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 18:21:04 2017

@author: kyleh
"""

import numpy as np
import scipy.io.arff as arff
import argparse
import time
import partition
import NeuralNetwork
import DataTransform
import evaluate
import preprocess

###get data from file
fileName='adult-big.arff'
f=open(fileName, 'r')
data, meta = arff.loadarff(f)
d = data.copy()

###get training parameters
parser = argparse.ArgumentParser()  
parser.add_argument('-e', '--epochs', help='set number of epochs', default=10)
parser.add_argument('-H', '--hidden', help='set number of hidden layer nodes', default=10)
parser.add_argument('-r', '--rate', help='set learning rate', default=.2)
args=parser.parse_args()
epochs=args.epochs
rate=args.rate
hidden=args.hidden
print('Running for',epochs,'epochs,',hidden,'hidden nodes, and with a learning rate of ',rate)

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

###get range of each attribute
ranges={}
for i,att in enumerate(meta.names()):
    if meta.types()[i]=='nominal':
        ranges[att]=list(set(d[att]))
    else:
        ranges[att]=['#']
#ranges['native-country']=[b'North-America', b'Central-South-America', b'Europe', b'Asia', b'Africa', b'Oceania']

start=time.time()
###preproces and train 10 models
models=[]
errors1=[]
errors2=[]
for i in range(len(partitions)):
    test=partitions.pop(0)
    testCopy=test.copy()
    train=np.concatenate(partitions).copy()
    print('Training Model '+str(i+1)+' ('+str(len(train))+' samples)')
    preprocess.z_score(train, meta)
#    print(train)
    nnData=DataTransform.transform(train, meta, ranges)
#    print(nnData)
    nn = NeuralNetwork.NeuralNetwork(epochs=epochs, hidden=hidden, learningRate=rate)
    errors=nn.train(nnData, meta)
    errors1.append(errors[0])
    errors2.append(errors[1])
    models.append(nn)
    partitions.append(test)
    print()
end=time.time()

print('Results:')
print('\tAverage error after 1 epoch=',sum(errors1)/len(errors1))
print('\tAverage error after last epoch=',sum(errors2)/len(errors2))

evaluators=[]
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
    evaluators.append(e)
#    print('Accuracy='+str(e.getAccuracy()))
#    print('Macro Precision=',e.macroPrecision)
#    print('Macro Recall=',e.macroRecall)
#    print('Macro F1=',e.macroF1)
#    print()
    
print('\tAverage Macro F1=',sum(ev.macroF1 for ev in evaluators)/len(evaluators))
print('\tRuntime=',end-start)