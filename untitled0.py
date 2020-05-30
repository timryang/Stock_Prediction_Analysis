# -*- coding: utf-8 -*-
"""
Created on Thu May 28 22:38:17 2020

@author: timot
"""


import matplotlib as plt
import numpy as np
xVals = np.arange(0,1,0.01)
yVals = np.arange(0,2,0.01)

matrix = (1/(xVals+1))*np.exp(-3*yVals[0]/(xVals+1))

for y in yVals[1:]:
    temp = (1/(xVals+1))*np.exp(-3*y/(xVals+1))
    matrix = np.vstack((temp, matrix))
    