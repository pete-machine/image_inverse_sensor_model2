#!/usr/bin/env python

import os
import cv2
import sys
import rospy
import numpy as np
from sensor_msgs.msg import Image, CameraInfo
from nav_msgs.msg import OccupancyGrid
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Pose, Point, Quaternion
from boundingbox_msgs.msg import ImageDetections
from image2ism_new import InversePerspectiveMapping
import message_filters

sys.path.append("/usr/lib/python2.7/dist-packages")
rospy.init_node('image2ism', anonymous=True)
nodeName = rospy.get_name()

# Topics IN
topicInMultiClass = rospy.get_param(nodeName+'/topicInMultiClass', '')
topicInSingleClass = rospy.get_param(nodeName+'/topicInSingleClass', '')
topicCamInfo = rospy.get_param(nodeName+'/topicCamInfo', nodeName+'UnknownInputTopic')

# Topics Out
topicOutPrefix = rospy.get_param(nodeName+'/topic_out_prefix', '/ism')
# Topics out is specified in a config file
configFile = rospy.get_param(nodeName+'/config_file', 'cfg/bb2ismExample.cfg')

# Setting match2Grid==False ignores min/max_likelihood. 
match2Grid = rospy.get_param(nodeName+'/match2Grid', True)

minLikelihood = rospy.get_param(nodeName+'/min_likelihood', 0.4) 
maxLikelihood = rospy.get_param(nodeName+'/max_likelihood', 0.8)

# Image size
resample_input = rospy.get_param(nodeName+'/resample_input', 1.0) 

# Grid settings
grid_resolution = rospy.get_param(nodeName+'/grid_resolution', 0.05) # 10*pi/180
degCutBelowHorizon = rospy.get_param(nodeName+'/degCutBelowHorizon', 7.5) # 10*pi/180


############## Intrinsic Camera settings ####################
# Use either the camera info topic to get intrinsic parameters (useCameraInfo== True).
useCameraInfo= rospy.get_param(nodeName+'/useCameraInfo', True) 

# Or specify all values explictly (useCameraInfo == False) 
if useCameraInfo == False:
    focal_length_x = rospy.get_param(nodeName+'/focal_length_x', 1.0) 
    focal_length_y = rospy.get_param(nodeName+'/focal_length_y', 1.0) 
    principal_point_offset_x = rospy.get_param(nodeName+'/principal_point_offset_x', 0.0) 
    principal_point_offset_y = rospy.get_param(nodeName+'/principal_point_offset_y', 0.0) 
    axis_skew = rospy.get_param(nodeName+'/axis_skew', 0.0) 
    org_image_dim_width = rospy.get_param(nodeName+'/org_image_dim_width', 800) 
    org_image_dim_height = rospy.get_param(nodeName+'/org_image_dim_height', 600) 

    # Focal length fx,fy
    focal_length = np.array([focal_length_x,focal_length_y]) # np.array([582.5641679547480,581.4956911617406])
    
    # Principal point offset x0,y0
    principal_point_offset = np.array([principal_point_offset_x,principal_point_offset_y]) # np.array([512.292836648199,251.580929633186])
    
    # Both the input resolution used when calibrating the camera and the resolution of the input in required. 
    # Image dimensions. (The input image size to be remapped and the original image size is not always the same !!! If e.g. the input image have been resized)
    org_image_dim = [org_image_dim_height, org_image_dim_width] #[544,1024] # The original image is used for in the intrinsic parameters to estimate image rays in world coordinates. 
###############################################################
    

############## Extrinsic Camera settings ####################
# Use either a tf-tree extrinsic camera settings (useTfForExtrinsic== True).
useTfForExtrinsic = False # THIS HAVEN'T BEEN IMPLEMENTED YET
targetFrameId = rospy.get_param(nodeName+'/targetFrameId', 'UnknownFrameId') 

# Or specify all values explictly (useTfForExtrinsic == False) 
if useTfForExtrinsic == False:
    cam_zTranslation = rospy.get_param(nodeName+'/cam_zTranslation', 1.5)
    cam_pitch = rospy.get_param(nodeName+'/cam_pitch', 0.0)  #20*np.pi/180
    cam_yaw = rospy.get_param(nodeName+'/cam_yaw', 0.0) # 10*pi/180
    cam_FOV = rospy.get_param(nodeName+'/cam_FOV', 0.349) # 20*pi/180
    
    degCutBelowHorizon = 7.5 # 10.0
    cam_roll = 0.0
    
    #pCamera = [cam_xTranslation,cam_yTranslation,cam_zTranslation]
    pCamera = [0,0,cam_zTranslation]
###############################################################

## THIS IS NOT IMPLEMENTED 
#ignoreRectangle = rospy.get_param(nodeName+'/ignore_rectangle', False)
#strIgnoreRectangleCoordinates = rospy.get_param(nodeName+'/ignore_rectangle_coordinates', "0.0 1.0 0.0 0.222") # normalized coordinates (rowMin, rowMax, colMin, colMax).

    
cam_info_sub = message_filters.Subscriber(topicCamInfo, CameraInfo)
if topicInSingleClass == '':
    det_image_sub = message_filters.Subscriber(topicInMultiClass, ImageDetections)
else:
    image_sub = message_filters.Subscriber(topicInSingleClass, Image)
    
#ipm = InversePerspectiveMapping(grid_resolution,degCutBelowHorizon)
ipm = InversePerspectiveMapping(grid_resolution,degCutBelowHorizon,minLikelihood,maxLikelihood,False)


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
## Functions to handle detection image with and without a camera info message. 
def callbackDetectionImageReceived_CameraInfo(data,info):
    if ipm.isIntrinsicUpdated == False:
        ipm.update_intrinsic_from_CameraInfo(info)
    baseDetectionImageReceived(data)
    
def callbackDetectionImageReceived_NoCameraInfo(data):
    if ipm.isIntrinsicUpdated == False:
        ipm.update_intrinsic(focal_length,principal_point_offset,org_image_dim,axis_skew)
    baseDetectionImageReceived(data)

def baseDetectionImageReceived(data):
    imgConfidence = bridge.imgmsg_to_cv2(data.imgConfidence, desired_encoding="passthrough")
    imgClass = bridge.imgmsg_to_cv2(data.imgClass, desired_encoding="passthrough")
    crop = data.crop
    
    if imgClass.shape[0] == 1 and imgClass.shape[1] == 1: 
        imgClass = imgClass[0,0]*np.ones(imgConfidence.shape,dtype=np.uint8)
    
    image2ism(imgConfidence,imgClass,data.header)
    
## Functions to handle an with and without a camera info message. 
def callbackSingleClass_CameraInfo(data,info):
    if ipm.isIntrinsicUpdated == False:
        ipm.update_intrinsic_from_CameraInfo(info)
    baseSingleClass(data)
    
def callbackSingleClass_NoCameraInfo(data):
    if ipm.isIntrinsicUpdated == False:
        ipm.update_intrinsic(focal_length,principal_point_offset,org_image_dim,axis_skew)
    baseSingleClass(data)
    
def baseSingleClass(data):
    imgConfidence = bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
    #print "len(imgConfidence.shape): ", len(imgConfidence.shape), "imgConfidence.shape: ", imgConfidence.shape
    if len(imgConfidence.shape) == 3: # To handle RGB images.
        imgConfidence = cv2.cvtColor(imgConfidence,cv2.COLOR_BGR2GRAY)
    classNumber = outputTopicsNumber[outputTopicsNumber.keys()[0]]
    imgClass = classNumber*np.ones(imgConfidence.shape)
    crop = np.array([0.0,1.0,0.0,1.0])
    
    image2ism(imgConfidence,imgClass,data.header)
    

def image2ism(imgConfidence,imgClass,header):
            
    if ipm.isExtrinsicUpdated == False:
        ipm.update_extrinsic(cam_pitch,cam_yaw,cam_roll,pCamera)    

    for objectType in pubOutputTopics.keys():
        classNumber = outputTopicsNumber[objectType]
        bwClass = imgClass==classNumber
        imgConfidenceClass = np.zeros_like(imgConfidence)
        imgConfidenceClass[bwClass] = imgConfidence[bwClass]
        
    
        resample_input
        if resample_input == 1.0:
            cv_image = imgConfidenceClass
        else:            
            cv_image = cv2.resize(imgConfidenceClass,None,fx=resample_input,fy=resample_input,interpolation=cv2.INTER_NEAREST)
        
        imDimIn = cv_image.shape[0:2]
        if ipm.isHomographyUpdated == False:
            pRayStarts,pDst,rHorizon, rHorizonTrue,pSrc,pDstOut = ipm.update_homography(imDimIn)

        # A grid is only published when the homography is updated.
        if ipm.isHomographyUpdated == True: 
            grid = ipm.makePerspectiveMapping(cv_image, match2Grid = match2Grid)
            
            grid_msg = OccupancyGrid()
            grid_msg.header = header
            #grid_msg.header.stamp = rospy.Time.now()
            grid_msg.header.frame_id = targetFrameId
            grid_msg.info.resolution = grid_resolution
            grid_msg.info.width = grid.shape[1]
            grid_msg.info.height = grid.shape[0]
            origin_x = ipm.distToMapping
            origin_y = -float(grid.shape[0])*grid_resolution/2
            origin_z = 0.0
            grid_msg.info.origin = Pose(Point(origin_x, origin_y, origin_z),Quaternion(0, 0, 0, 1))
            grid_msg.data = np.flipud(grid).flatten()
            
            pubOutputTopics[objectType].publish(grid_msg)



#cam_info_sub = message_filters.Subscriber(topicCamInfo, CameraInfo)
#det_image_sub = message_filters.Subscriber(topicInMultiClass, ImageDetections)
#image_sub = message_filters.Subscriber(topicInSingleClass, Image)

# main
def main():
    print ''
    if topicInMultiClass is not '':
        if useCameraInfo : 
            msgFilter0 = message_filters.TimeSynchronizer([det_image_sub,cam_info_sub], 10)
            msgFilter0.registerCallback(callbackDetectionImageReceived_CameraInfo)
            print 'image2ism (', nodeName, '): ', 'is subscriping to topic: ', topicInMultiClass, '(DetectionImage) and ', topicCamInfo 
        else:
            rospy.Subscriber(topicInMultiClass, ImageDetections, callbackDetectionImageReceived_NoCameraInfo, queue_size=1)
            print 'image2ism (', nodeName, '): ', 'is subscriping to topic: ', topicInMultiClass, '(DetectionImage)'
        
    
    if topicInSingleClass is not '':
        if useCameraInfo : 
            msgFilter0 = message_filters.TimeSynchronizer([image_sub,cam_info_sub], 10)
            msgFilter0.registerCallback(callbackSingleClass_CameraInfo)
            print 'image2ism (', nodeName, '): ', 'is subscriping to topic: ', topicInSingleClass, '(Image) and ', topicCamInfo
        else:
            rospy.Subscriber(topicInSingleClass, Image, callbackSingleClass_NoCameraInfo, queue_size=1)
    
    for className in pubOutputTopics.keys():
        print 'image2ism (', nodeName, ') is publishing: ', os.path.join(topicOutPrefix,className), "Class Number: ", outputTopicsNumber[className]
    
    rospy.spin()


if __name__ == '__main__':
    main()
