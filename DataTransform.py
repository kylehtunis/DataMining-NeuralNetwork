# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 18:23:50 2017

@author: kyleh
"""

import numpy as np
import pandas

def transform(data, meta, ranges):
    
    tdata={}
    
    for i, att in enumerate(meta.names()):
        if meta.types()[i]=='nominal':
#            print(pandas.get_dummies(data[att]).as_matrix()[0])
            tdata[att]=pandas.get_dummies(data[att])
            missing_cols=set(ranges[att])-set(data[att])
            for col in missing_cols:
#                print(col)
                tdata[att][col] = 0
                print('Data transform resulted in an incomplete set, added "',att,'" to complete the set')
            tdata[att]=tdata[att].as_matrix()
#            print(tdata)
        else:
            tdata[att]=[[data[att][i]] for i in range(len(data[att]))]
    
#    print(tdata)
    return tdata

def getGoldLabels(data, meta, ranges):
    tdata=transform(data, meta, ranges)
    labels = [np.argmax(tdata[meta.names()[-1]][i]) for i in range(len(data))]
    return labels