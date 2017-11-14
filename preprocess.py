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
        if b'?' in l:
            classCounts[l.index(b'?')][:]=0
        for i in range(0,n):
            if data[i][j]==b'?':
#                print(data[i][j])
                c=data[i][-1]
                data[i][j]=l[classCounts[:,classes.index(c)].tolist().index(max(classCounts[:,classes.index(c)]))]
                count+=1
    print('Replaced '+str(count)+' missing values.\n')
#    return data
    
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

def groupByContinent(data):
    asia=[b'Hong', b'Vietnam', b'Thailand', b'India', b'China', b'Japan', b'Laos', b'Philippines', b'Taiwan', b'Iran', b'Cambodia']
    na=[b'Trinadad&Tobago', b'Cuba', b'Jamaica', b'Puerto-Rico', b'United-States', b'Haiti', b'Outlying-US(Guam-USVI-etc)', b'Canada', b'Dominican-Republic']
    csa=[b'Guatemala', b'Peru', b'Honduras', b'Ecuador', b'Mexico', b'El-Salvador', b'Nicaragua', b'Columbia']
    europe=[b'Ireland', b'Germany', b'Scotland', b'France', b'Holand-Netherlands', b'Hungary', b'Portugal', b'Yugoslavia', b'England', b'Greece', b'Italy', b'Poland']
    for i, sample in enumerate(data):
        if sample['native-country'] in asia:
            sample['native-country']=b'Asia'
        elif sample['native-country'] in csa:
            sample['native-country']=b'Central-South-America'
        elif sample['native-country'] in na:
            sample['native-country']=b'North-America'
        elif sample['native-country'] in europe:
            sample['native-country']=b'Europe'
#    return data

def groupEducation(data):
    shs=[b'11th', b'9th', b'12th', b'12th']
    nhs=[b'1st-4th', b'5th-6th', 'Preschool', b'7th-8th']
    for sample in data:
        if sample['education'] in shs:
            sample['education']=b'Some-high-school'
        elif sample['education'] in nhs:
            sample['education']=b'No-high-school'
            
def groupMarried(data):
    married=[b'Married-spouse-absent', b'Married-AF-spouse', b'Married-civ-spouse']
    for sample in data:
        if sample['marital-status'] in married:
            sample['marital-status']=b'Married'
        elif sample['marital-status'] != b'Never-married':
            sample['marital-status']='Formerly-married'
        