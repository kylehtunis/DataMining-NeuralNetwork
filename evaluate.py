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
#                print(self.pred[i])
#                print(self.gold[i])
                correct+=1
        return correct/self.size
    
    def confusionMatrices(self):
        self.matrices={l:[0,0,0,0] for l in set(self.gold)} #tp, fp, fn, tn
        for i in range(self.size):
#            print(self.gold[i])
#            print(self.pred[i], '\n')
            if self.pred[i]==self.gold[i]:
                self.matrices[self.pred[i]][0]+=1
                for l in set(self.gold):
                    self.matrices[l][3]+=1
                self.matrices[self.pred[i]][3]-=1
            else:
                self.matrices[self.gold[i]][2]+=1
                self.matrices[self.pred[i]][1]+=1
#            print(self.matrices)
                
    def measures(self):
#        try:
#            self.precision={l:self.matrices[l][0]/(self.matrices[l][0]+self.matrices[l][1]) for l in set(self.gold)}
#        except ZeroDivisionError:
#            self.precision={l:0 for l in set(self.gold)}
#        finally:
#            try:
#                self.recall={l:self.matrices[l][0]/(self.matrices[l][0]+self.matrices[l][2]) for l in set(self.gold)}
#            except ZeroDivisionError:
#                self.recall={l:0 for l in set(self.gold)}
#            finally:
#                try:
#                    self.f1={l:2*self.precision[l]*self.recall[l]/(self.precision[l]+self.recall[l]) for l in set(self.gold)}
#                except ZeroDivisionError:
#                    self.f1=0
#                finally:
        self.precision={}
        self.recall={}
        self.f1={}
        for l in set(self.gold):
            try:
                self.precision[l]=self.matrices[l][0]/(self.matrices[l][0]+self.matrices[l][1])
            except ZeroDivisionError:
                self.precision[l]=0
            finally:
                try:
                    self.recall[l]=self.matrices[l][0]/(self.matrices[l][0]+self.matrices[l][2])
                except ZeroDivisionError:
                    self.recall[l]=0
                finally:
                    try:
                        self.f1[l]=2*self.precision[l]*self.recall[l]/(self.precision[l]+self.recall[l])
                    except ZeroDivisionError:
                        self.f1[l]=0
        self.macroPrecision=sum(self.precision[l] for l in set(self.gold))/2
        self.macroRecall=sum(self.recall[l] for l in set(self.gold))/2
        self.macroF1=(self.macroPrecision+self.macroRecall)/2