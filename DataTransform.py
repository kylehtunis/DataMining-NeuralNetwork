# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 18:23:50 2017

@author: kyleh
"""

import numpy as np
import pandas

def transform(data, meta):
    
    tdata={}
    
    for i, att in enumerate(meta.names()):
        if meta.types()[i]=='nominal':
#            print(pandas.get_dummies(data[att]).as_matrix()[0])
            tdata[att]=pandas.get_dummies(data[att]).as_matrix()
        else:
            tdata[att]=[[data[att][i]] for i in range(len(data[att]))]
    
#    print(tdata)
    return tdata

def getGoldLabels(data, meta):
    tdata=transform(data, meta)
    labels = [tdata[meta.names()[-1]][i] for i in range(len(data))]
    return labels