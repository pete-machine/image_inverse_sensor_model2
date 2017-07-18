#!/usr/bin/env python

import os
import cv2
import sys
import rospy
import numpy as np
from sensor_msgs.msg import Image
from nav_msgs.msg import OccupancyGrid
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Pose, Point, Quaternion
from ismFunctions import inversePerspectiveMapping, image2ogm
from boundingbox_msgs.msg import ImageDetections
from image2ism_new import InversePerspectiveMapping

sys.path.append("/usr/lib/python2.7/dist-packages")
rospy.init_node('image2ism', anonymous=True)
nodeName = rospy.get_name()
#print "NodeName:", nodeName
#objectTypeInt = rospy.get_param(nodeName+'/objectTypeInt', 1000) # 1000 is not specified. 0-19 is pascal classes. 20 is the pedestrian detector
#topicsInName = rospy.get_param(nodeName+'/topicIns', 'UnknownInputTopic') # 1000 is not specified. 0-19 is pascal classes. 20 is the pedestrian detector
# Topics
topicInMultiClass = rospy.get_param(nodeName+'/topicInMultiClass', '')
topicInSingleClass = rospy.get_param(nodeName+'/topicInSingleClass', '')

if topicInMultiClass == '' and topicInSingleClass == '':
    raise NameError('Either topicInMultiClass or topicInSingleClass should be defined in the launch script') 
    
topicOutPrefix = rospy.get_param(nodeName+'/topic_out_prefix', '/ism')
configFile = rospy.get_param(nodeName+'/config_file', 'cfg/bb2ismExample.cfg')

# Image size
imageWidth = rospy.get_param(nodeName+'/imageWidth', 800) 
imageHeight = rospy.get_param(nodeName+'/imageHeight', 600) 

# Camera Settings
#cam_xTranslation = rospy.get_param(nodeName+'/cam_xTranslation', 0) 
#cam_yTranslation = rospy.get_param(nodeName+'/cam_yTranslation', 0) 
cam_zTranslation = rospy.get_param(nodeName+'/cam_zTranslation', 1.5)
cam_pitch = rospy.get_param(nodeName+'/cam_pitch', 0.349)  #20*np.pi/180
cam_yaw = rospy.get_param(nodeName+'/cam_yaw', 0.1745) # 10*pi/180
cam_FOV = rospy.get_param(nodeName+'/cam_FOV', 0.349) # 20*pi/180
targetFrameId = rospy.get_param(nodeName+'/targetFrameId', 'UnknownFrameId') 

# Grid settings
grid_resolution = rospy.get_param(nodeName+'/grid_resolution', 0.05) # 10*pi/180
#grid_xSizeInM = rospy.get_param(nodeName+'/grid_xSizeInM', -1.0) # For values <0 length of X is scaled automatically
#grid_ySizeInM = rospy.get_param(nodeName+'/grid_ySizeInM', -1.0) # For values <0 length of Y is scaled automatically    
minLikelihood = rospy.get_param(nodeName+'/min_likelihood', 0.4) 
maxLikelihood = rospy.get_param(nodeName+'/max_likelihood', 0.8)
ignoreRectangle = rospy.get_param(nodeName+'/ignore_rectangle', False)
strIgnoreRectangleCoordinates = rospy.get_param(nodeName+'/ignore_rectangle_coordinates', "0.0 1.0 0.0 0.222") # normalized coordinates (rowMin, rowMax, colMin, colMax).

#############################################################
## Intrinsic Camera settings
# Focal length fx,fy
fl = np.array([582.5641679547480,581.4956911617406])

# Principal point offset x0,y0
ppo = np.array([512.292836648199,251.580929633186])

# Axis skew
s = 0.0

degCutBelowHorizon = 7.5 # 10.0
cam_roll = 0

# Both the input resolution used when calibrating the camera and the resolution of the input in required. 
# Image dimensions. (The input image size to be remapped and the original image size is not always the same !!! If e.g. the input image have been resized)
imDimOrg = [544,1024] # The original image is used for in the intrinsic parameters to estimate image rays in world coordinates. 
imDimIn = [imageHeight,imageWidth] # The input dimensions

#pCamera = [cam_xTranslation,cam_yTranslation,cam_zTranslation]
pCamera = [0,0,cam_zTranslation]
degCutBelowHorizon = 7.5

ipm = InversePerspectiveMapping(grid_resolution,degCutBelowHorizon)
ipm.update_intrinsic(fl,ppo,imDimOrg)
ipm.update_extrinsic(cam_pitch,cam_pitch,cam_roll,pCamera)
pRayStarts,pDst,rHorizon, rHorizonTrue,pSrc,pDstOut = ipm.update_homography(imDimIn)

#points = np.vstack((pCamera,pRayStarts,pDst))
#############################################################




#ignoreRectangleCoord = [float(strPartCoord) for strPartCoord in strIgnoreRectangleCoordinates.split(' ')] 
#(Xvis, Yvis,rHorizon) = inversePerspectiveMapping(imageWidth, imageHeight, cam_xTranslation, cam_yTranslation, cam_zTranslation, cam_pitch, cam_yaw, cam_FOV);

configData = open(configFile,'r') 
configText = configData.read()
strsClassNumberAndName = [line for idx,line in enumerate(str(configText).split('\n')) if line is not '' and idx is not 0]
pubOutputTopics = {}
outputTopicsNumber = {}

for strClass in strsClassNumberAndName:
    strNumberAndClass = strClass.split(' ')
    
    topicOutName = os.path.join(topicOutPrefix,strNumberAndClass[1])
    #print('Class: ',  int(strNumberAndClass[0]), ', ObjectType: ',  strNumberAndClass[1], ', outputTopicName: ', topicOutName)
    outputTopicsNumber[strNumberAndClass[1]] = int(strNumberAndClass[0])
    # Class: Names are used in dictonary
    pubOutputTopics[strNumberAndClass[1]] = rospy.Publisher(topicOutName, OccupancyGrid, queue_size=1)
    

    
bridge = CvBridge()



vectorLength = 6
def callbackDetectionImageReceived(data):
    imgConfidence = bridge.imgmsg_to_cv2(data.imgConfidence, desired_encoding="passthrough")
    imgClass = bridge.imgmsg_to_cv2(data.imgClass, desired_encoding="passthrough")
    crop = data.crop
    
    if imgClass.shape[0] == 1 and imgClass.shape[1] == 1: 
        imgClass = imgClass[0,0]*np.ones(imgConfidence.shape,dtype=np.uint8)
    
    image2ism(imgConfidence,imgClass)
    
    
def callbackDetectionImageReceivedSingleClass(data):
    imgConfidence = bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
    classNumber = outputTopicsNumber[outputTopicsNumber.keys()[0]]
    imgClass = classNumber*np.ones(imgConfidence.shape)
    crop = np.array([0.0,1.0,0.0,1.0])
    
    image2ism(imgConfidence,imgClass)
    
    
    
def image2ism(imgConfidence,imgClass):
    
    
    #for each class
    for objectType in pubOutputTopics.keys():
        classNumber = outputTopicsNumber[objectType]
        bwClass = imgClass==classNumber
        imgConfidenceClass = np.zeros_like(imgConfidence)
        imgConfidenceClass[bwClass] = imgConfidence[bwClass]
        
#        print "np.unique(imgConfidenceClass)",np.unique(imgClass),"imgClass.shape",imgClass.shape, "classNumber: ", classNumber
#        print "np.min(imgConfidenceClass)",np.min(imgConfidenceClass),"np.max(imgConfidenceClass)",np.max(imgConfidenceClass)
        
        cv_image = cv2.resize(imgConfidenceClass,(imageWidth, imageHeight)).astype(np.int)
        
        
        grid = ipm.makePerspectiveMapping(cv_image)
        
        # Convert to grid format (0-100)
        grid = grid.astype(np.float32)
        mask = grid>0
        grid[mask] = grid[mask]*(maxLikelihood-minLikelihood)/255.0+minLikelihood
        grid[mask==False] = 0.5
        grid = (grid*100).astype(np.uint8)

        grid_msg = OccupancyGrid()
        grid_msg.header.stamp = rospy.Time.now()
        grid_msg.header.frame_id = targetFrameId
        grid_msg.info.resolution = grid_resolution
        grid_msg.info.width = grid.shape[1]
        grid_msg.info.height = grid.shape[0]
    #    origin_x = dist_x1+cam_xTranslation;
    #    origin_y = dist_y1+cam_yTranslation;
    #    print dist_x1, dist_y1
        origin_x = 0
        origin_y = 0
        origin_z = 0
        grid_msg.info.origin = Pose(Point(origin_x, origin_y, origin_z),Quaternion(0, 0, 0, 1))
        grid_msg.data = grid.flatten()
        
        #print "np.min(grid)",np.min(grid),"np.max(grid)",np.max(grid)
        # print "Sequence value", data.header.frame_id
        # HACK: LOADS THE OBJECT TYPE FROM THE FRAME ID:
        #pubImageObjs[0].publish(grid_msg)
        
        pubOutputTopics[objectType].publish(grid_msg)
        #pubImageObjsDictionary[data.header.frame_id].publish(grid_msg)
    
#vectorLength = 6
#def callbackDetectionImageReceived(data):
#    cv_image = bridge.imgmsg_to_cv2(data, "mono8").astype(float)
#    cv_image = cv2.resize(cv_image,(imageWidth, imageHeight))
#
#    # Hack
#    strParts = data.header.frame_id.split('/')
#    objectType = strParts[-1]
#
#    if ignoreRectangle:
#        setIndices = (np.array([cv_image.shape[0],cv_image.shape[0],cv_image.shape[1],cv_image.shape[1]])*np.array(ignoreRectangleCoord)).astype(int)
#        cv_image[setIndices[0]:setIndices[1],setIndices[2]:setIndices[3]] = -10
#
#
#    if objectType == 'human':
#        objectExtent = 0.4
#        #print(cv_image)
#    elif objectType == 'other':
#        objectExtent = 0.5
#    elif objectType == 'unknown':
#        objectExtent = 0.5
#    elif objectType == 'vehicle':
#        objectExtent = 1.5
#    elif objectType == 'water':
#        objectExtent = 0.0
#    elif objectType == 'grass':
#        objectExtent = 0.0
#    elif objectType == 'ground':
#        objectExtent = 0.0
#    elif objectType == 'shelterbelt':
#        objectExtent = 0.0
#    elif objectType == 'anomaly':
#        objectExtent = 0.5
#    elif objectType == 'heat':
#        objectExtent = 0.5   
#    else:
#        objectExtent = 0.0
#        
#    #,ignoreRectangle,ignoreRectangleCoord
#    #(Xvis, Yvis) = inversePerspectiveMapping(cv_image.shape[1], cv_image.shape[0], rHorizon, 0, 0, 1.5, 20*np.pi/180, 10*np.pi/180, 20*np.pi/180);
#    #print objectType, objectExtent
#    #tStart = time.time()
#    grid, nGridX, nGridY, dist_x1, dist_y1,empty = image2ogm(Xvis,Yvis,cv_image,rHorizon,grid_xSizeInM,grid_ySizeInM,grid_resolution,objectExtent,minLikelihood,maxLikelihood)
#    #print "Elapsed on Image2ism: ", time.time()-tStart
#    #image_message = bridge.cv2_to_imgmsg(grid, encoding="mono8")
#    #pubImage.publish(image_message)
#    
#    grid_msg = OccupancyGrid()
#    grid_msg.header.stamp = rospy.Time.now()
#    grid_msg.header.frame_id = "base_link_mount" 
#    grid_msg.info.resolution = grid_resolution
#    grid_msg.info.width = nGridX
#    grid_msg.info.height = nGridY
##    origin_x = dist_x1+cam_xTranslation;
##    origin_y = dist_y1+cam_yTranslation;
##    print dist_x1, dist_y1
#    origin_x = dist_x1
#    origin_y = dist_y1
#    origin_z = 0;    
#    grid_msg.info.origin = Pose(Point(origin_x, origin_y, origin_z),Quaternion(0, 0, 0, 1))
#    grid_msg.data = grid.flatten()
#    # print "Sequence value", data.header.frame_id
#    # HACK: LOADS THE OBJECT TYPE FROM THE FRAME ID:
#    #pubImageObjs[0].publish(grid_msg)
#
#    pubImageObjsDictionary[data.header.frame_id].publish(grid_msg)



# main
def main():
    print ''
    if topicInSingleClass == '':
        rospy.Subscriber(topicInMultiClass, ImageDetections, callbackDetectionImageReceived, queue_size=1)
        print 'image2ism (', nodeName, '): ', 'is subscriping to topic: ', topicInMultiClass
    else:
        rospy.Subscriber(topicInSingleClass, Image, callbackDetectionImageReceivedSingleClass, queue_size=1)
        print 'image2ism (', nodeName, '): ', 'is subscriping to topic: ', topicInSingleClass
    
    for className in pubOutputTopics.keys():
        print 'image2ism (', nodeName, ') is publishing: ', os.path.join(topicOutPrefix,className), "Class Number: ", outputTopicsNumber[className]
    
    rospy.spin()


if __name__ == '__main__':
    main()
