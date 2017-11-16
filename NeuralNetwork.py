# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 18:23:04 2017

@author: kyleh
"""

import numpy as np
import DataTransform

class NeuralNetwork:
    
    
    def __init__(self, **kwargs):
        if 'hidden' not in kwargs.keys():
            kwargs['hidden']=10
        if 'learningRate' not in kwargs.keys():
            kwargs['learningRate']=.1
        if 'epochs' not in kwargs.keys():
            kwargs['epochs']=100
        if 'minError' not in kwargs.keys():
            kwargs['minError']=0
        self.hidden=kwargs['hidden']
        self.learningRate=kwargs['learningRate']
        self.epochs=kwargs['epochs']
        self.minError=kwargs['minError']
        self.bh=np.random.rand(kwargs['hidden'],1)/5
        
        
    def train(self, data, meta):
        ###initialize weight vectors
        self.inputsize=sum(len(data[att][0]) for att in meta.names()[:-1])
#        print(inputsize)
        self.Wh=np.random.rand(self.inputsize,self.hidden)
        self.outputsize=len(data[meta.names()[-1]][0])
        self.bo=np.random.rand(self.outputsize,1)
        self.Wo=np.random.rand(self.hidden,self.outputsize)
        for epoch in range(self.epochs):
            avgErr=0.
            learningRate=1
            if self.learningRate==-1 and epoch<=100:
                learningRate=1./np.sqrt(epoch+1)
            else:
                learningRate=self.learningRate
#            print('Epoch '+str(epoch+1)+'\r')
            for i in range(len(data[meta.names()[0]])):
                ###get sample
                Oi=[]
                for t, att in enumerate(meta.names()[:-1]):
                    if meta.types()[t]=='nominal':
                        Oi+=data[att][i].tolist()
                    else:
                        Oi+=data[att][i]
                pred, Oh=self.classifySample(np.array(Oi).reshape((len(Oi),1)))
#                print(Oh)
    #            pred=pred.tolist()
#                print(pred)
                gold=data[meta.names()[-1]][i].tolist()
#                print(gold)
                
                ###calculate errors and update weights
                oErr=[pred[j]*(1-pred[j])*(gold[j]-pred[j]) for j in range(self.outputsize)]
#                print(str(oErr)+'\n')
                for j in range(self.outputsize):
                    for i in range(self.hidden):
                        self.Wo[i][j]+=learningRate*oErr[j]*Oh[i]
                    self.bo[j]+=learningRate*oErr[j]
                hErr=[Oh[j]*(1-Oh[j])*sum(oErr[k]*self.Wo[j][k] for k in range(len(oErr))) for j in range(self.hidden)]
                for j in range(self.hidden):
                    for i in range(self.inputsize):
                        self.Wh[i][j]+=learningRate*hErr[j]*Oi[i]
                    self.bh[j]+=learningRate*hErr[j]
#            print(oErr)
                avgErr+=sum(abs(e) for e  in oErr)/len(data[meta.names()[0]])
            print('AvgErr: ', avgErr)
            if avgErr<=self.minError:
                print('Reached min error after ',epoch+1,' epochs')
                epoch=self.epochs
                break
            if epoch==0:
#                print('AvgErr after 1 epoch:', avgErr)
                err1=avgErr
#            avgErr=0
#        if epoch==self.epochs-1:
#            print('Reached maximum epochs')
        err2=avgErr
#        print('AvgErr after last epoch: ', avgErr)
        return err1, err2
    
    def classify(self, data, meta):
#        nnData=DataTransform.transform(data, meta)
        results=[]
        for i in range(len(data[meta.names()[0]])):
            nnSample=[]
            for t, att in enumerate(meta.names()[:-1]):
                if meta.types()[t]=='nominal':
                    nnSample+=data[att][i].tolist()
                else:
                    nnSample+=data[att][i]
#            print(nnSample)
            classification=self.classifySample(np.array(nnSample).reshape(len(nnSample),1))[0]
#            print(classification)
            results.append(np.argmax(classification))
        return results
        
    def classifySample(self, sample):
        ###calculate hidden layer outputs
        Wht=np.transpose(self.Wh)
#        print(Wht.shape)
#        print(sample.shape)
        Oh=np.array([self.sigmoid(np.dot(Wht,sample)[i]+self.bh[i]) for i in range(self.hidden)])
#        print(Oh.shape)

        ###calculate output layer outpus
        Wot=np.transpose(self.Wo)
#        print(Wot.shape)
        Oo=[self.sigmoid(np.dot(Wot[i],Oh)+self.bo[i])[0] for i in range(self.outputsize)]
#        print(Oo)
        return Oo, [element[0] for element in Oh]
        
    def sigmoid(self, val):
        return 1./(1+np.e**-val)