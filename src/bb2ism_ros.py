#!/usr/bin/env python

import os 
import rospy
import numpy as np
from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import Pose, Point, Quaternion, PointStamped
from nav_msgs.msg import OccupancyGrid
#from sensor_msgs.msg import PointCloud
import matplotlib.pyplot as plt
import tf2_ros
import tf2_geometry_msgs
#from tf2_ros import BufferInterface
#from transforms3d.euler import euler2mat, mat2euler, quat2mat

from bb2ism import Bb2ism
#from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud

#bf = BufferInterface()

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
baseFrameId = rospy.get_param(nodeName+'/base_frame_id', 'UnknownFrameId') 

# Less important values
degradOutlook = rospy.get_param(nodeName+'/degrad_outlook', True)
degradOutlookAfterM = rospy.get_param(nodeName+'/degrad_outlook_afterM', 10.0)
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
    print('Class: ',  int(strNumberAndClass[0]), ', ObjectType: ',  strNumberAndClass[1], ', outputTopicName: ', topicOutName)
    pubOutputTopics[int(strNumberAndClass[0])] = rospy.Publisher(topicOutName, OccupancyGrid, queue_size=1)



xyz_conf_empty = {key: [] for key in pubOutputTopics.keys()}

bb2ism = Bb2ism(hFOV, localizationErrorStd,localizationErrorStdEnd, degradOutlook, degradOutlookAfterM, max_distance,grid_resolution,pVisible,pNonVisible,pMaxLikelyhood)


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
    
    print("Marker Received")
    for marker in markerArray.markers:
        if marker.action is marker.DELETEALL:
            cHeader = marker.header
            print("    Empty")
        else:
            print("    Filled")
            
            pt = PointStamped()
            pt.header = marker.header
            pt.point = marker.pose.position

            #print("Out pt: ", pt.point)        
            try:
                # Bug-fix. To remove the first '/' in frame. E.g. '/Multisensor/blah' --> 'Multisensor/blah' 
                strParts = marker.header.frame_id.split('/')
                if strParts[0] is '':
                    headFrame = str.join('/',strParts[1:])
                else:
                    headFrame = marker.header.frame_id
                #print "headFrame", headFrame, "baseFrameId",baseFrameId
                trans = tfBuffer.lookup_transform(baseFrameId,headFrame, rospy.Time())
                
                pt = tf2_geometry_msgs.do_transform_point(pt, trans)
                #trans = tfBuffer.lookup_transform( 'Multisense/left_camera_optical_frame','velodyne', rospy.Time())
                #pose.orientation = trans.transform.rotation
                #pt = tfBuffer.transform(pt,baseFrameId)
                #print("pose.orientation:",pose.orientation)
                xyz_conf[marker.id].append(np.array([pt.point.x,pt.point.y,pt.point.z,marker.color.a]))
            except Exception as e: #(tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                print("In bb2ism_ros node. No transform is found. Except: ",e.message,e.args)
                #raise NameError('In bb2ism_ros node. No transform is found.')
                
                #pose.orientation.w = 1
                #pass
            #print("Out pt: ", pt.point)

        
    #pubOutputTopics.publish()
    if cHeader is []:
        raise NameError('bb2ism_ros node expect for each received marker array with marker.header and marker.action.DELETEALL')
        
    for key in pubOutputTopics.keys():
        grid = bb2ism.drawDetections(xyz_conf[key])
        
        grid_msg = OccupancyGrid()
        grid_msg.header = cHeader
        #grid_msg.header.stamp = rospy.Time.now()
        grid_msg.header.frame_id = baseFrameId
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
        
        
        
    print xyz_conf
    
rospy.Subscriber(topicIn, MarkerArray, callback_bbReceived,queue_size=None)  
# main
def main():
    
    rospy.spin()

if __name__ == '__main__':
    main()
    
    
#    def markerArray2safeObjectArray(markerArray):
#    safeObjects = SafeObjectArray()
#    for marker in markerArray.markers:
#        # For visualization a dummie markers is created with a deleteall-type. This marker needs to be skipped.
#        if marker.action is not marker.DELETEALL:
#            safeObjects.safe_objects.append(copy.deepcopy(marker2safeObject(marker)))
#    return safeObjects
#
#
#    def marker2safeObject(marker_in):
#    safe_object = SafeObject()
#    
#    safe_object.header = marker_in.header
#    safe_object.id = marker_in.id
#
#    safe_object.type = marker_in.ns
#
#    safe_object.det_confidence_level = marker_in.color.a
#    safe_object.obj_orientation.orientation = marker_in.pose.orientation
#    safe_object.obj_orientation.quality = 0
#
#    safe_object.obj_position.x = marker_in.pose.position.x
#    safe_object.obj_position.y = marker_in.pose.position.y
#    safe_object.obj_position.z = marker_in.pose.position.z
#    safe_object.obj_position.quality = 0
#    
#    safe_object.obj_lin_vel.x = 0
#    safe_object.obj_lin_vel.y = 0
#    safe_object.obj_lin_vel.z = 0
#    safe_object.obj_lin_vel.quality = 0
#    
#    safe_object.obj_size.x = marker_in.scale.x
#    safe_object.obj_size.y = marker_in.scale.y
#    safe_object.obj_size.z = marker_in.scale.z
#    safe_object.obj_size.quality = 0
#    
#    return safe_object