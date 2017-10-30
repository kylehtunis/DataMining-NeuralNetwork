# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 22:50:38 2017

@author: kyleh
"""

class Evaluator:
    
    def __init__(self, gold, pred):
        self.gold=gold
        self.pred=pred
        self.size=len(pred)
        
    def getAccuracy(self):
        correct=0
        for i in range(self.size):
            if self.pred[i]==self.gold[i]:
                correct+=1
        return correct/self.size
    
    def confusionMatrices(self):
        self.matrices={l:[0,0,0,0] for l in set(self.gold)} #tp, fp, fn, tn
        for i in range(self.size):
            if self.pred[i]==self.gold[i]:
                self.matrices[self.pred[i]][0]+=1
                for l in set(self.gold):
                    self.matrices[l][3]+=1
                self.matrices[self.pred[i]][3]-=1
            else:
                self.matrices[self.gold[i]][2]+=1
                self.matrices[self.pred[i]][1]+=1
                
    def measures(self):
        self.precision={l:self.matrices[l][0]/(self.matrices[l][0]+self.matrices[l][1]) for l in set(self.gold)}
        self.recall={l:self.matrices[l][0]/(self.matrices[l][0]+self.matrices[l][2]) for l in set(self.gold)}
        self.f1={l:2*self.precision[l]*self.recall[l]/(self.precision[l]+self.recall[l]) for l in set(self.gold)}