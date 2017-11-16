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
epochs=int(args.epochs)
rate=float(args.rate)
hidden=int(args.hidden)
print('Running for',epochs,'epochs,',hidden,'hidden nodes, and with a learning rate of',rate)

print('\nPreprocessing\n')
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
    preprocess.z_score(train, meta)
#    print(train)
    print('Transforming Data for Model',i+1)
    nnData, gold=DataTransform.transform(train, meta, ranges)
#    print(nnData)
    print('Training Model '+str(i+1)+' ('+str(len(train))+' samples)')
    nn = NeuralNetwork.NeuralNetwork(epochs=epochs, hidden=hidden, learningRate=rate)
    nn.train(nnData, gold, meta)
    models.append(nn)
    partitions.append(test)
    print()
end=time.time()

print('Results:')
print('\tAverage error after 1 epoch=',[sum(abs(err) for err in errs)/len(models) for errs in zip(*(m.err1 for m in models))])
print('\tAverage error after last epoch=',[sum(abs(err) for err in errs)/len(models) for errs in zip(*(m.errLast for m in models))])

evaluators=[]
###evaluate results
for i, model in enumerate(models):
    test=partitions[i].copy()
    preprocess.z_score(test, meta)
    nnData, goldLabels=DataTransform.transform(test, meta, ranges)
    #    print(nnData)
    results=model.classify(nnData, meta)
    #    print(results)
    goldLabels=[np.argmax(l) for l in goldLabels]
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