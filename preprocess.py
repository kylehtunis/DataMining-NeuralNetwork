# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 18:21:55 2017

@author: kyleh
"""

import numpy as np

def missing_values(data, meta):
    
    
    #######useful variables
    n=len(data)
    classLabel=meta.names()[-1]
    classes=list(set(data[classLabel]))
    numOfAtts=len(meta.names())
    m=len(set(data[classLabel]))
    
    #######fix missing values
    count=0
    for j in range(0,numOfAtts):
        classCounts=np.ndarray((len(set(data[meta.names()[j]])),m))
        classCounts[:][:]=0
        l=set(data[meta.names()[j]])
        l=list(l)
        for k in range(0, n):
            classCounts[l.index(data[k][j])][classes.index(data[k][-1])]+=1
        for i in range(0,n):
            if data[i][j]==b'?':
                c=data[i][-1]
                data[i][j]=l[classCounts[classes.index(c)].tolist().index(max(classCounts[classes.index(c)][:]))]
                count+=1
    print('Replaced '+str(count)+' missing values.')
    
def z_score(data, meta): 
    
    count=0
    for i in range(len(meta.names())):
        if meta.types()[i]=='nominal':
            continue
        std = np.std(data[meta.names()[i]])
        mean = np.mean(data[meta.names()[i]])
        for sample in data:
            sample[i]=(sample[i]-mean)/std
        count+=1
#    print('Replaced values with z-scores for '+str(count)+' attributes.')