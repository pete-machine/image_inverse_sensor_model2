#!/usr/bin/env python

import rospy
import time
from collections import namedtuple
from std_msgs.msg import Float64MultiArray
from std_msgs.msg import UInt16
import sys
sys.path.append("/usr/lib/python2.7/dist-packages")
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import numpy as np

#from sound_play.libsoundplay import SoundClient

#distanceBoundary1 = 1
#distanceBoundary2 = 3
#distanceBoundary3 = 5
#timeBetweenEvaluation = 0.2 # Time between evaulations in seconds
#timeBetweenVoice = 3.0 # The  robot must repeat state every X seconds
#time2ForgetHumans = 3.0 # Forget humans after X seconds!
#Human = namedtuple('Human', ['dist-ance', 'angle','timeSinceDetection'])
# OBJECT TYPES (YOLO)
# 0.aeroplane, 1. bicycle, 2 bird, 3.boat, 4. bottle, 5. bus, 6. car, 7. cat, 8. chair, 9. cow, 10.diningtable
# 11. dog, 12. horse, 13. motorbike, 14. person, 15. pottedplant, 16. sheep, 17. sofa, 18. train, 19. tvmonitor, 20. pedestrian
# REMAPPING (agriculture classes)

##unknown = [];                  #0
#animal = [2,7,9,11,12,16];      #1
##building = [];                 #2
##field = [];                    #3
##ground = [];                   #4
#obstacle = [4,8,10,15,17,19];   #5
#person = [14];                  #6
##shelterbelt = [];              #7
##sky = [];                      #8
#vehicle = [0,1,3,5,6,13,18];    #9
##water = [];                    #10
rospy.init_node('bb2image', anonymous=True)
nodeName = rospy.get_name()
topicInName  = rospy.get_param(nodeName+'/topicInName', '/bbUnknownObjects')
#topicOutName = rospy.get_param('/image_inverse_sensor_model/topicOutName', '/detImageUnknown')
#objectTypeInt = rospy.get_param(nodeName+'/objectTypeInt', 1000) # 1000 is not specified. 0-19 is pascal classes. 20 is the pedestrian detector
imgDimWidth  = rospy.get_param(nodeName+'/imgDimWidth', 800)
imgDimHeight = rospy.get_param(nodeName+'/imgDimHeight', 600)

#def numbers_to_strings(argument):
#    switcher = {
#        0: "unknown",
#        1: "vehicle",
#        2: "human",
#    }
#    return switcher.get(argument, "Unknown")
def numbers_to_strings(argument):
    switcher = {
        0: "human",
        1: "vehicle",
        2: "animal",
		3: "obstacle",
    }
    return switcher.get(argument, "Unknown")


objectTypes =  np.array([False, False, False, False])
for iObj in range(0,len(objectTypes)):
    objectTypes[iObj] = rospy.get_param(nodeName+'/objectType_'+numbers_to_strings(iObj), False)


strParts = topicInName.split('/')
topicOutNameBase = '/det/' + strParts[3] + '/' + strParts[1] + '/'

pubImageObjs = list()
topicOutNames = list()
for iType in range(0,len(objectTypes)):
    topicOutNames.append(topicOutNameBase + numbers_to_strings(iType))
    pubImageObjs.append(rospy.Publisher(topicOutNames[-1], Image , queue_size=1))
    


#print topicOutName
#pubImage = rospy.Publisher(topicOutName, Image , queue_size=1)
bridge = CvBridge()

vectorLength = 6
def callbackBB_received(data):
    #print("bb_received")
    blank_images = list()
    for cObjectType in range(0,len(objectTypes)):
        blank_images.append(np.zeros((imgDimHeight,imgDimWidth,1), np.uint8))
    #print data.data
    
    
    for iObject in range(0,len(data.data)/vectorLength):
        cBasePoint = vectorLength*iObject
        cObjectType = int(data.data[cBasePoint+5])
        #print iObject, cBasePoint
        if (objectTypes[cObjectType]): 
            pt1 = (int(data.data[cBasePoint+0]*imgDimWidth),int(data.data[cBasePoint+1]*imgDimHeight))
            pt2 = (int((data.data[cBasePoint+0]+data.data[cBasePoint+2])*imgDimWidth),int((data.data[cBasePoint+1]+data.data[cBasePoint+3])*imgDimHeight))
            cv2.rectangle(blank_images[cObjectType],pt1,pt2,int(data.data[cBasePoint+4]*255),-1)        
    
    for cObjectType in range(0,len(objectTypes)):
        #print 'cObjectType', cObjectType, 'objectTypes[cObjectType]', objectTypes[cObjectType]
        if (objectTypes[cObjectType]): 
            image_message = bridge.cv2_to_imgmsg(blank_images[cObjectType], encoding="mono8")
            image_message.header.frame_id = topicOutNames[cObjectType]
            pubImageObjs[cObjectType].publish(image_message)



# main
def main():
    
    for iType in range(0,len(objectTypes)):
        if(objectTypes[iType]==True):
            print 'SemanticSegmentation  publishing:"', topicOutNameBase + numbers_to_strings(iType), ', receiving:"', topicInName
    
    rospy.Subscriber(topicInName, Float64MultiArray, callbackBB_received,queue_size=1)    
    #rospy.Timer(rospy.Duration(timeBetweenEvaluation), EvaluateHumanAwareness)
    rospy.spin()


if __name__ == '__main__':
    main()
