#!/usr/bin/env python

import os 
import rospy
import numpy as np
from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import Pose, Point, Quaternion, PointStamped
from nav_msgs.msg import OccupancyGrid
#import matplotlib.pyplot as plt
import tf2_ros
import tf2_geometry_msgs
from bb2ism import Bb2ism


rospy.init_node('marker_array_2_msg_safe', anonymous=False)
nodeName = rospy.get_name()

# Important values (Set in launch file to match the given case)
topicIn = rospy.get_param(nodeName+'/topic_in', '/ImageBBox3d')
configFile = rospy.get_param(nodeName+'/config_file', 'cfg/bb2ismExample.cfg')
topicOutPrefix = rospy.get_param(nodeName+'/topic_out_prefix', '/ism')
grid_resolution = rospy.get_param(nodeName+'/grid_resolution', 0.1)
hFOV = rospy.get_param(nodeName+'/cam_horisontal_FOV', np.pi/2)
strLocalizationErrorStd = rospy.get_param(nodeName+'/localization_error_std', '0.8 4.0') # Two floats seperated by space.
strLocalizationErrorStdEnd = rospy.get_param(nodeName+'/localization_error_std_end', strLocalizationErrorStd) # Two floats seperated by space.
localizationErrorStd = np.array([float(value) for value in strLocalizationErrorStd.split(' ')])
localizationErrorStdEnd = np.array([float(value) for value in strLocalizationErrorStdEnd.split(' ')])


pVisible = rospy.get_param(nodeName+'/p_visible', 0.4)
pMaxLikelyhood = rospy.get_param(nodeName+'/p_max_likelyhood', 0.8)
max_distance = rospy.get_param(nodeName+'/max_distance', 50.0)
targetFrameId = rospy.get_param(nodeName+'/base_frame_id', 'UnknownFrameId') 

# Less important values
degradeOutlook = rospy.get_param(nodeName+'/degrade_outlook', True)
degradeOutlookAfterM = rospy.get_param(nodeName+'/degrade_outlook_afterM', 10.0)
pNonVisible = rospy.get_param(nodeName+'/p_non_visible', 0.5)


tfBuffer = tf2_ros.Buffer()
listener = tf2_ros.TransformListener(tfBuffer)
pose = Pose()

configData = open(configFile,'r') 
configText = configData.read()
strsClassNumberAndName = [line for idx,line in enumerate(str(configText).split('\n')) if line is not '' and idx is not 0]
pubOutputTopics = {}

for strClass in strsClassNumberAndName:
    strNumberAndClass = strClass.split(' ')
    
    topicOutName = os.path.join(topicOutPrefix,strNumberAndClass[1])
    #print('Class: ',  int(strNumberAndClass[0]), ', ObjectType: ',  strNumberAndClass[1], ', outputTopicName: ', topicOutName)
    
    # Class: Names are used in dictonary
    pubOutputTopics[strNumberAndClass[1]] = rospy.Publisher(topicOutName, OccupancyGrid, queue_size=1)



xyz_conf_empty = {key: [] for key in pubOutputTopics.keys()}

bb2ism = Bb2ism(hFOV, localizationErrorStd,localizationErrorStdEnd, degradeOutlook, degradeOutlookAfterM, max_distance,grid_resolution,pVisible,pNonVisible,pMaxLikelyhood)


## Small test
#xys = np.array([[17,5],[14,12],[25,-10],[29,8]])
#detectionGrid = bb2ism.drawDetections(xys)
#plt.figure()
#plt.imshow(detectionGrid)
#plt.axis('equal')
#plt.colorbar()
#plt.show()


def callback_bbReceived(markerArray):
    xyz_conf = {key: [] for key in pubOutputTopics.keys()}
    cHeader = []
    
    for marker in markerArray.markers:
        if marker.action is marker.DELETEALL:
            cHeader = marker.header
        else:
            pt = PointStamped()
            pt.header = marker.header
            pt.point = marker.pose.position


            # Bug-fix. To remove the first '/' in frame. E.g. '/Multisensor/blah' --> 'Multisensor/blah' 
            strParts = marker.header.frame_id.split('/')
            if strParts[0] is '':
                cCameraFrame = str.join('/',strParts[1:])
            else:
                cCameraFrame = marker.header.frame_id
            validTransform = True
            
            # Get transformation. If get transformation fails do noting.
            try:                
                # Get transform                
                trans = tfBuffer.lookup_transform(targetFrameId,cCameraFrame, rospy.Time()) #marker.header.stamp) 
                
                # Transform camera point to 
                pt = tf2_geometry_msgs.do_transform_point(pt, trans)
                                
            except Exception as e: #(tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                print("In bb2ism_ros node. No transform is found. Except: ",e.message,e.args)
                validTransform = False
            
            
            if validTransform == True:
                # Get class type from namespace
                strClassName = marker.ns.split('/')[-1]
                #print "marker.ns: ", marker.ns, "strClassName: ", strClassName, "xyz_conf.keys(): ", xyz_conf
                # Append point to a dictionary-class.
                xyz_conf[strClassName].append(np.array([pt.point.x,pt.point.y,marker.color.a]))
                
            

        
    #pubOutputTopics.publish()
    if cHeader is []:
        raise NameError('bb2ism_ros node expect for each received marker array with marker.header and marker.action.DELETEALL')
        
    for key in pubOutputTopics.keys():
        grid = bb2ism.drawDetections(xyz_conf[key])
        
        grid_msg = OccupancyGrid()
        grid_msg.header = cHeader
        #grid_msg.header.stamp = rospy.Time.now()
        grid_msg.header.frame_id = targetFrameId
        grid_msg.info.resolution = bb2ism.grid_resolution
        grid_msg.info.width = bb2ism.gridWidth
        grid_msg.info.height = bb2ism.gridHeight
    
        origin_x = bb2ism.gridOriginXYZ[0] #dist_x1
        origin_y = bb2ism.gridOriginXYZ[1] #dist_y1
        origin_z = bb2ism.gridOriginXYZ[2]

        #cloud_out = do_transform_cloud(cloud_in, trans)
        
        grid_msg.info.origin = Pose(Point(origin_x, origin_y, origin_z),Quaternion(0, 0, 0, 1))
        grid_msg.data = (grid*100.0).astype(int).flatten()
        
        pubOutputTopics[key].publish(grid_msg)

    
rospy.Subscriber(topicIn, MarkerArray, callback_bbReceived,queue_size=None)  
# main
def main():
    print ''
    print 'bb2ism (', nodeName, ') is subscriping to topic: ', topicIn
    for className in pubOutputTopics.keys():
        print 'bb2ism (', nodeName, ') is publishing: ', os.path.join(topicOutPrefix,className), "Class Number: ", pubOutputTopics[className]
    
    
    rospy.spin()

if __name__ == '__main__':
    main()
    
    
