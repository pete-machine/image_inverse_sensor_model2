#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 15 15:18:58 2017

@author: pistol
"""
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import cv2
import time 

# intersection function
def isect_line_plane(p0, p1, p_co, p_no, epsilon=1e-6):
    """
    p0, p1: define the line
    p_co, p_no: define the plane:
        p_co is a point on the plane (plane coordinate).
        p_no is a normal vector defining the plane direction;
             (does not need to be normalized).

    return a Vector or None (when the intersection can't be found).
    """
    u = p1-p0
    dot = np.dot(u,p_no)
    
    w = p0-p_co
    fac = -np.dot(p_no, w) / dot
    u = u*np.expand_dims(fac,axis=1)
    out = p0+u
    
    # fac: > 0.0 behind p0 (bad case)
    # fac: < 1.0 infront of p1 (Good case)
    # If abs(dot) is small. Lines are parallel?
    return out, fac, abs(dot)

#
#    u = p1-p0
#    dot = np.dot(p_no, u)
#
#    if abs(dot) > epsilon:
#        # the factor of the point between p0 -> p1 (0 - 1)
#        # if 'fac' is between (0 - 1) the point intersects with the segment.
#        # otherwise:
#        #  < 0.0: behind p0.
#        #  > 1.0: infront of p1.
#        w = p0-p_co
#        fac = -np.dot(p_no, w) / dot
#        u = u*fac
#        out = p0+u
#        return out
#    else:
#        # The segment is parallel to plane
#        return None
    
t1 = time.time()
camHeight = 2.0;

# Define line by two points P0 (cam position). P1 (Pixel point)
lineP0 = np.array([0,0,camHeight],dtype=np.double)
lineP1 = np.array([1,0.1,1.8],dtype=np.double)

# Focal length fx,fy
fl = np.array([582.5641679547480,581.4956911617406])

# Principal point offset x0,y0
ppo = np.array([512.292836648199,251.580929633186])

# Axis skew
s = 0.0


## INIT stuff
# Make image points from camera matrix (in a way to complicated way). 
imDim = [544,1024]

radFOV = 2*np.arctan(imDim/(2*fl))

## For new transform
# Make transoformation for camera. 
degYaw = 0
#degPitch = 19.0 # Degrees in downward direction.
degPitch = 20.0 # Degrees in downward direction.
degRoll = 0

radYaw = degYaw*np.pi/180
radPitch = degPitch*np.pi/180
radRoll = degRoll*np.pi/180

# Estimate horison only from pitch
#rHorizonTrue = imDim[0]*(radFOV[0]/2-radPitch)
rHorizonTrue = int(np.ceil( (imDim[0]-1)/2*(1 - np.tan(radPitch)/np.tan(radFOV[0]/2)) + 1 ));
cutBelowHorizon = 10*np.pi/180 # 5 degrees
#rHorizon = imDim[0]*(radFOV[0]/2+cutBelowHorizon-radPitch)
rHorizon =  int(np.ceil( (imDim[0]-1)/2*(1 - np.tan(radPitch-cutBelowHorizon)/np.tan(radFOV[0]/2)) + 1 ));


print 'rHorizonTrue: ', rHorizonTrue, 'rHorizon: ', rHorizon


rHorizon = np.maximum(rHorizon,0)
rHorizonTrue = np.maximum(rHorizonTrue,0)

#imDim[0] = imDim[0]-rHorizon
imCorners = np.array([[0,0],[0,imDim[0]-rHorizon],[imDim[1],imDim[0]-rHorizon],[imDim[1],0]])
imCorners = np.hstack((imCorners,np.ones((imCorners.shape[0],1))))
pSrc = imCorners
K = np.array([[fl[0], s, ppo[0]],[0,fl[1],ppo[1]],[0,0,1]])

Kinv = np.linalg.inv(K)
camCorners = np.matmul(Kinv,imCorners.T)
camCorners = np.vstack((camCorners[2,:],camCorners[0,:],camCorners[1,:]))
#pSrc = camCorners


# Define transformation.
Ryaw = np.array([[np.cos(radYaw), -np.sin(radYaw),0],[np.sin(radYaw),np.cos(radYaw),0],[0,0,1]])
Rpitch = np.array([[np.cos(radPitch),0,np.sin(radPitch)],[0,1,0],[-np.sin(radPitch),0,np.cos(radPitch)]])
Rroll = np.array([[1,0,0],[0,np.cos(radRoll),-np.sin(radRoll)],[0,np.sin(radRoll),np.cos(radRoll)]])
Rtrans = np.matmul(Rroll,np.matmul(Rpitch,Ryaw))
# Transform extreem points of image 
#camCorners = np.matmul(Rroll,np.matmul(Rpitch,np.matmul(Ryaw,camCorners)))
camCorners = np.matmul(Rtrans,camCorners)
linePX = lineP0+1.0*camCorners.transpose()

print linePX
# Define plane by point and normal vector.
planeP = np.array([0,0,0],dtype=np.double)
planeN = np.array([0,0,1],dtype=np.double)

pDst,fac,val = isect_line_plane(lineP0,linePX,planeP,planeN)
#print pDst
#print fac
#print val 

rgb = cv2.cvtColor(cv2.imread('/home/pistol/DataFolder/stereo2.png'),cv2.COLOR_BGR2RGB)

## Define output
pDstOut = pDst*10
pDstOut = pDstOut-np.min(pDstOut,axis=0)
pDstSize = np.max(pDstOut,axis=0).astype(np.int)


M = cv2.getPerspectiveTransform(pSrc[:,:2].astype(np.float32),pDstOut[:,:2].astype(np.float32))
#M,ret = cv2.findHomography(pSrc[1:,:].transpose(),pDst[:,:2],0)

print "fac: ", fac
#print "out: ", out

#t1 = time.time()

warped = cv2.warpPerspective(np.flipud(rgb[rHorizon:,:,:]), M,(pDstSize[0],pDstSize[1]) , flags=cv2.INTER_LINEAR)
print 'run time: ', (time.time()-t1)/100,  's' 
#rgb = cv2.line(rgb,np.array([0, 100]),np.array([0,0]),np.array([255,255,255]),5)

if rHorizonTrue > 0:
    rgb[int(rHorizonTrue):int(rHorizonTrue)+3,:,:] = [255,0,0]
    rgb[int(rHorizon):int(rHorizon)+3,:,:] = [0,0,255]


plt.figure()
plt.imshow(rgb)
plt.figure()
plt.imshow(warped)


#fig = plt.figure()
#
#ax = fig.add_subplot(111, projection='3d')
#n = 100
#
#points = np.vstack((lineP0,linePX,pDst))
#X = points[:,0]
#Y = points[:,1]
#Z = points[:,2]
#ax.scatter(X,Y,Z)
#ax.set_xlabel('X Label')
#ax.set_ylabel('Y Label')
#ax.set_zlabel('Z Label')
#
#max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0
#
#mid_x = (X.max()+X.min()) * 0.5
#mid_y = (Y.max()+Y.min()) * 0.5
#mid_z = (Z.max()+Z.min()) * 0.5
#ax.set_xlim(mid_x - max_range, mid_x + max_range)
#ax.set_ylim(mid_y - max_range, mid_y + max_range)
##ax.set_zlim(mid_z - max_range, mid_z + max_range)
#ax.set_zlim(0, mid_z + max_range)
#plt.show()