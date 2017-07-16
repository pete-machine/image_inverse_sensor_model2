
import time
import sys
import numpy as np
import numpy.matlib as ml
from PIL import Image

import sys
sys.path.append("/usr/lib/python2.7/dist-packages")
import cv2

from skimage.draw import polygon
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
rHorizon = 0

# intersection function
def intersection_line_plane(p0, p1, p_co, p_no, epsilon=1e-6):
    """
    p0, p1: define the line
    p_co, p_no: define the plane:
        p_co is a point on the plane (plane coordinate).
        p_no is a normal vector defining the plane direction;
             (does not need to be normalized).

    return a Vector
    """
    u = p1-p0
    dot = np.dot(u,p_no)
    
    w = p0-p_co
    fac = -np.dot(p_no, w) / dot
    u = u*np.expand_dims(fac,axis=1)
    out = p0+u
    
    # fac: > 0.0 behind p0 (bad case)
    # fac: < 1.0 infront of p1 (Good case)
    # If abs(dot) is small - lines are parallel. 
    return out, fac, abs(dot)

def determineHorizon(imDim,radFOV,radPitch, degCutBelowHorizon=10.0):
    # Estimate horison only from pitch
    #rHorizonTrue = imDim[0]*(radFOV[0]/2-radPitch)
    rHorizonTrue = int(np.ceil( (imDim[0]-1)/2*(1 - np.tan(radPitch)/np.tan(radFOV[0]/2)) + 1 ));
    cutBelowHorizon = degCutBelowHorizon*np.pi/180 # 5 degrees
    #rHorizon = imDim[0]*(radFOV[0]/2+cutBelowHorizon-radPitch)
    rHorizon =  int(np.ceil( (imDim[0]-1)/2*(1 - np.tan(radPitch-cutBelowHorizon)/np.tan(radFOV[0]/2)) + 1 ));
    
    
    print 'rHorizonTrue: ', rHorizonTrue, 'rHorizon: ', rHorizon
    
    rHorizon = np.maximum(rHorizon,0)
    rHorizonTrue = np.maximum(rHorizonTrue,0)
    
    return  rHorizon, rHorizonTrue

## Image points to camera frame (transformation based on intrinsic parameters)
#def invIntrinsicCameraTransform(fl,ppo,s imDim,rHorizon):
#    
#    # Axis skew
#    s = 0.0
#    
#
#    K = np.array([[fl[0], s, ppo[0]],[0,fl[1],ppo[1]],[0,0,1]])
#    Kinv = np.linalg.inv(K)
#    
#    return Kinv, radFOV

# Angling of camera: pitch, yaw, roll of
# Camera position: pCam = [x,y,z] = [x,y,height]
def extrinsicTransform(pitch,yaw,roll, pCamera, inDeg = False):
    
    # In degree convert to radians
    if inDeg:
        pitch = pitch*np.pi/180
        yaw = yaw*np.pi/180
        roll = roll*np.pi/180
        
    #translation = [0,0,cameraHeight]
#    # Define transformation.
#    T_pitch = np.array([[np.cos(pitch),0,np.sin(pitch)],[0,1,0],[-np.sin(pitch),0,np.cos(pitch)]])
#    T_yaw = np.array([[np.cos(yaw), -np.sin(yaw),0],[np.sin(yaw),np.cos(yaw),0],[0,0,1]])
#    T_roll = np.array([[1,0,0],[0,np.cos(roll),-np.sin(roll)],[0,np.sin(roll),np.cos(roll)]])
    
    T_pitch = np.array([[np.cos(pitch),0,np.sin(pitch),0],[0,1,0,0],[-np.sin(pitch),0,np.cos(pitch),0],[0,0,0,1]])
    T_yaw = np.array([[np.cos(yaw), -np.sin(yaw),0,0],[np.sin(yaw),np.cos(yaw),0,0],[0,0,1,0],[0,0,0,1]])
    T_roll = np.array([[1,0,0,0],[0,np.cos(roll),-np.sin(roll),0],[0,np.sin(roll),np.cos(roll),0],[0,0,0,1]])


    T = np.matmul(T_roll,np.matmul(T_pitch,T_yaw))
    T[0:3,3] = pCamera 
    
#    T_translation = np.eye(4)
#    T_translation[0:3,3] = pCamera 
#    T = np.matmul(T_translation,np.matmul(T_roll,np.matmul(T_pitch,T_yaw)))
    

    return T
    
    
#def blah(imDim):
#    
#    imCorners = np.array([[0,0],[0,imDim[0]-rHorizon],[imDim[1],imDim[0]-rHorizon],[imDim[1],0]])
#    imCorners = np.hstack((imCorners,np.ones((imCorners.shape[0],1))))
#    pSrc = imCorners
#    
#    
#    
#    camCorners = np.matmul(Kinv,imCorners.T)
#    camCorners = np.vstack((camCorners[2,:],camCorners[0,:],camCorners[1,:]))
#    
#    #pSrc = camCorners
#    
#    
#    
#    # Transform extreem points of image 
#    #camCorners = np.matmul(Rroll,np.matmul(Rpitch,np.matmul(Ryaw,camCorners)))
#    camCorners = np.matmul(Rtrans,camCorners)
#    linePX = lineP0+1.0*camCorners.transpose()
#    
#    print linePX
##    # Define plane by point and normal vector.
##    planeP = np.array([0,0,0],dtype=np.double)
##    planeN = np.array([0,0,1],dtype=np.double)
##    
##    pDst,fac,val = intersection_line_plane(lineP0,linePX,planeP,planeN)
#    
#
#    
#    return pDst,