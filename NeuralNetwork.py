# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 18:23:04 2017

@author: kyleh
"""

import numpy as np

class NeuralNetwork:
    
    
    def __init__(self, **kwargs):
        if 'hidden' not in kwargs.keys():
            kwargs['hidden']=10
        self.hidden=kwargs['hidden']
        self.bh=np.random.rand(kwargs['hidden'],1)/5
        
    def train(self, data, meta):
        inputsize=sum(len(data[att][0]) for att in meta.names()[:-1])
#        print(inputsize)
        self.Wh=np.random.rand(inputsize,self.hidden)
        outputsize=len(data[meta.names()[-1]][0])
        self.bo=np.random.rand(outputsize,1)
        self.Wo=np.random.rand(self.hidden,outputsize)
    
    def classify(self, data):
        return