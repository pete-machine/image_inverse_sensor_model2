#!/usr/bin/env python

import sys
sys.path.append("/usr/lib/python2.7/dist-packages")

#import rospy
import time
from collections import namedtuple

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import cv2
import numpy as np
from ismFunctions import inversePerspectiveMapping, image2ogm, TransformImageAccordingToObjectExtend

#rHorizonRatio = 500/size(anomaly,1);
#rHorizon = round(size(anomaly,1)*rHorizonRatio);


#inversePerspectiveMapping(size(anomaly,2), size(anomaly,1), rHorizon, 0, 0, 1.5, 25*pi/180, 10*pi/180, 20*pi/180);

objectExtentInput = 1

cv_image = cv2.imread('./tmpImage4.png',0)
(Xvis, Yvis,rHorizon) = inversePerspectiveMapping(cv_image.shape[1], cv_image.shape[0],  0, 0, 2.056, 0.1963, 0.0, 0.68067);

grid_xSizeInM = -1
grid_ySizeInM = -1
grid_resolution = 0.2
objectExtent = 2
grid, nGridX, nGridY, dist_x1, dist_y1,IcroppedTransformed = image2ogm(Xvis,Yvis,cv_image,rHorizon,grid_xSizeInM,grid_ySizeInM,grid_resolution,objectExtent)
        
plt.figure()
plt.imshow(cv_image,cmap='gray')
plt.figure()
plt.imshow(IcroppedTransformed,cmap='gray')
plt.figure()
plt.imshow(grid,cmap='gray')
cv_image[rHorizon-1:rHorizon+1,:] = 255
cv2.imwrite( 'Org' +".png", cv_image);
cv2.imwrite( 'Transformed' +".png", IcroppedTransformed);
cv2.imwrite( 'GridMap' +".png", grid*2.55);