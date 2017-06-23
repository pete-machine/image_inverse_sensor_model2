#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 10:41:39 2017

@author: pistol
"""


import time
import numpy as np
import matplotlib.pyplot as plt
from bb2ism import Bb2ism
import configparser as cp
import os

#import transforms3d
#import pymel.core.datatypes as dt

max_distance = 30.0 # max distance
hFOV =  np.pi*82.62/180 #np.pi/4 # Horisontal field of view. # 
grid_resolution = 0.5
pVisible = 0.4
pNonVisible = 0.5
pMaxLikelyhood = 0.8
degradeOutlook = True
degradeOutlookAfterM = 10.0 # distance
localizationErrorStd = np.array([.1, 0.8]) # 
localizationErrorStdEnd = np.array([.4, 2.0]) # 


import sys


def dVars(*strExpressions):
    frame = sys._getframe(1)
    for strExp in strExpressions:
        print strExp, ": ",  frame.f_globals[strExp]
        
    #dictOut = {strExp : frame.f_globals[strExp] for strExp in strExpressions}
    #print dictOut


#print(debug(bar))



###############################################################################

# Make readable config-file from configData-dict
cd = cp.ConfigParser()
cd.read("")

X = np.array([0,max_distance])
#Y0 = np.array([localizationErrorStd[0],localizationErrorStdEnd[0]])
#Y1 = np.array([localizationErrorStd[1],localizationErrorStdEnd[1]])

aStd = np.array([np.diff([localizationErrorStd[0],localizationErrorStdEnd[0]])/np.diff(X),np.diff([localizationErrorStd[1],localizationErrorStdEnd[1]])/np.diff(X)])
bStd = np.expand_dims(localizationErrorStd,1)

aStd*30.0+bStd
# 
xyz = [[4,1,1.0],[17,10,0.5],[12,-5,1.0],[31,8,1.0],[25,-21,0.5],[40.9702127, -0.44210202, 1.0]]
#xyz = []
#xyz = np.array([[7,2],[16,12],[25,-10],[29,8]])

t0 = time.time()

bb2ism = Bb2ism(hFOV, localizationErrorStd,localizationErrorStdEnd, degradeOutlook, degradeOutlookAfterM, max_distance,grid_resolution,pVisible,pNonVisible,pMaxLikelyhood)

t1 = time.time()

detectionGrid = bb2ism.drawDetections(xyz)

t2 = time.time()
    
print('Time (init): ', t1-t0)
print('Time (for ' + str(bb2ism.nDetections) + ' detections): ', (t2-t1))
print('Time (Total): ', t2-t0)

plt.figure()
plt.imshow(detectionGrid)
plt.axis('equal')
plt.colorbar()
plt.show()


## 
#xys = np.array([[7,2],[16,12],[25,-10],[25,8]])
#detectionGrid = bb2ism.drawDetections(xys)
#
#plt.figure()
#plt.imshow(detectionGrid)
#plt.axis('equal')
#plt.colorbar()
