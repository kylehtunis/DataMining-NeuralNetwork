# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 23:33:52 2017

@author: kyleh
"""

import numpy

def partition(data, splits):
    return numpy.array_split(data, splits)
