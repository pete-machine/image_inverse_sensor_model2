#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  2 10:37:27 2017

@author: pistol
"""
import cv2
import sys
import numpy as np
useOld = False

#if useOld:
#    from ismFunctions_old import inversePerspectiveMapping, image2ogm
#else:
from ismFunctions import inversePerspectiveMapping, image2ogm
import matplotlib.pyplot as plt
sys.path.append("/usr/lib/python2.7/dist-packages")

#topicOutPrefix = 
configFile = '../cfg/image2ismFcn8.cfg'

# Image size
imageWidth = 512 #rospy.get_param(nodeName+'/imageWidth', 800) 
imageHeight = 217 # rospy.get_param(nodeName+'/imageHeight', 600) 

# Camera Settings
cam_xTranslation = 0.0#rospy.get_param(nodeName+'/cam_xTranslation', 0) 
cam_yTranslation = 0.0#rospy.get_param(nodeName+'/cam_yTranslation', 0) 
cam_zTranslation = 2.056#rospy.get_param(nodeName+'/cam_zTranslation', 1.5)
cam_pitch = 0.3839#rospy.get_param(nodeName+'/cam_pitch', 0.349)  #20*np.pi/180
#cam_pitch = np.pi/5 # 
cam_yaw = 0.0 #rospy.get_param(nodeName+'/cam_yaw', 0.1745) # 10*pi/180
cam_rool = 0.0 
cam_FOV = 0.7835#rospy.get_param(nodeName+'/cam_FOV', 0.349) # 20*pi/180

# Grid settings
grid_resolution = 0.5 #rospy.get_param(nodeName+'/grid_resolution', 0.05) # 10*pi/180
grid_xSizeInM = -1.0 # rospy.get_param(nodeName+'/grid_xSizeInM', -1.0) # For values <0 length of X is scaled automatically
grid_ySizeInM = -1.0 # rospy.get_param(nodeName+'/grid_ySizeInM', -1.0) # For values <0 length of Y is scaled automatically    
minLikelihood = 0.4 # rospy.get_param(nodeName+'/min_likelihood', 0.4) 
maxLikelihood = 0.8 # rospy.get_param(nodeName+'/max_likelihood', 0.8)
ignoreRectangle = False # rospy.get_param(nodeName+'/ignore_rectangle', False)
strIgnoreRectangleCoordinates = '0.0 1.0 0.0 0.0' # rospy.get_param(nodeName+'/ignore_rectangle_coordinates', "0.0 1.0 0.0 0.222") # normalized coordinates (rowMin, rowMax, colMin, colMax).

ignoreRectangleCoord = [float(strPartCoord) for strPartCoord in strIgnoreRectangleCoordinates.split(' ')] 
(Xvis, Yvis,rHorizon) = inversePerspectiveMapping(imageWidth, imageHeight, cam_xTranslation, cam_yTranslation, cam_zTranslation, cam_pitch, cam_yaw, cam_FOV);

configData = open(configFile,'r') 
configText = configData.read()
strsClassNumberAndName = [line for idx,line in enumerate(str(configText).split('\n')) if line is not '' and idx is not 0]
pubOutputTopics = {}


outputTopicsNumber = {}
loadFile = False
if loadFile:
    for strClass in strsClassNumberAndName:
        strNumberAndClass = strClass.split(' ')
        outputTopicsNumber[strNumberAndClass[1]] = int(strNumberAndClass[0])
else:
    outputTopicsNumber['human'] = 0
#    outputTopicsNumber['other'] = 1
#    outputTopicsNumber['unknown'] = 2
#    outputTopicsNumber['building'] = 3
#    outputTopicsNumber['grass'] = 4
#    outputTopicsNumber['ground'] = 5
#    outputTopicsNumber['shelterbelt'] = 6
#    outputTopicsNumber['water'] = 7
    


filename = '/home/pistol/Code/ros_workspaces/private/src/fcn8_ros/src/semanticOutputData.npz'
filenameImage = '/home/pistol/Code/ros_workspaces/private/src/fcn8_ros/src/test_img2.png'
imgRaw = cv2.imread(filenameImage)

out = np.load(filename)
imgClass = out['imgClass']
imgConfidence = out['imgConfidence']



#for each class
for objectType in outputTopicsNumber.keys():
    classNumber = outputTopicsNumber[objectType]
    bwClass = imgClass==classNumber
    imgConfidenceClass = np.zeros_like(imgConfidence)
    imgConfidenceClass[bwClass] = imgConfidence[bwClass]
    
    cv_image = cv2.resize(imgConfidenceClass,(imageWidth, imageHeight)).astype(np.int)
    
    if ignoreRectangle:
        setIndices = (np.array([cv_image.shape[0],cv_image.shape[0],cv_image.shape[1],cv_image.shape[1]])*np.array(ignoreRectangleCoord)).astype(int)
        cv_image[setIndices[0]:setIndices[1],setIndices[2]:setIndices[3]] = -10

    if objectType == 'human':
        objectExtent = 0.4
    elif objectType == 'other':
        objectExtent = 0.5
    elif objectType == 'unknown':
        objectExtent = 0.5
    elif objectType == 'vehicle':
        objectExtent = 1.5
    elif objectType == 'water':
        objectExtent = 0.0
    elif objectType == 'grass':
        objectExtent = 0.0
    elif objectType == 'ground':
        objectExtent = 0.0
    elif objectType == 'shelterbelt':
        objectExtent = 0.0
    elif objectType == 'anomaly':
        objectExtent = 0.5
    elif objectType == 'heat':
        objectExtent = 0.5   
    else:
        objectExtent = 0.0

    grid, nGridX, nGridY, dist_x1, dist_y1,empty = image2ogm(Xvis,Yvis,cv_image,rHorizon,grid_xSizeInM,grid_ySizeInM,grid_resolution,objectExtent,minLikelihood,maxLikelihood)
    plt.figure()
    plt.imshow(imgRaw)
    plt.plot([0,imgRaw.shape[1]],[rHorizon,rHorizon])
    
    plt.figure()
    plt.imshow(grid)
    plt.axis('equal')
