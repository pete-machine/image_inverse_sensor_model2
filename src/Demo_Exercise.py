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
from image2ism_new import extrinsicTransform, determineHorizon,intersection_line_plane

rgb = cv2.cvtColor(cv2.imread('/home/pistol/DataFolder/stereo2.png'),cv2.COLOR_BGR2RGB)
t1 = time.time()



# SETTINGS

## Extrinsic Camera settings
# Camera angling
degPitch = 20.0 # Degrees in downward direction.
degYaw = 0
degRoll = 0

# Camera position 
camHeight = 2.0;
pCamera = np.array([0,0,camHeight])

## Intrinsic Camera settings
# Focal length fx,fy
fl = np.array([582.5641679547480,581.4956911617406])

# Principal point offset x0,y0
ppo = np.array([512.292836648199,251.580929633186])

# Axis skew
s = 0.0

# Image dimensions
imDim = [544,1024]

# Grid resolution. 
resolution = 0.1


# Convert to radians
radPitch = degPitch*np.pi/180
radYaw = degYaw*np.pi/180
radRoll = degRoll*np.pi/180

# Determine field-of-view from focal length and image dimensions.
radFOV = 2*np.arctan(imDim/(2*fl))

# The horizon is determined. 
# rHorizonTrue: The actual horizon. 
# rHorizon: To avoid uncertain localization, the image is cropped below the horizon by degCutBelowHorizon
rHorizon, rHorizonTrue = determineHorizon(imDim,radFOV,radPitch,degCutBelowHorizon=10)

# Camera intrinsic matrix is determined. 
K = np.array([[fl[0], s, ppo[0]],[0,fl[1],ppo[1]],[0,0,1]])
Kinv = np.linalg.inv(K)


# Image corners in matrix. 
imCorners = np.array([[0,0],[0,imDim[0]-rHorizon],[imDim[1],imDim[0]-rHorizon],[imDim[1],0]])
imCorners = np.hstack((imCorners,np.ones((imCorners.shape[0],1))))
pSrc = imCorners

# image corners are converted to image space. 
camCorners = np.matmul(Kinv,imCorners.T)

# Flip axis and make coordinates 3D homogeneous. 
camCorners = np.vstack((camCorners[2,:],camCorners[0,:],camCorners[1,:],np.ones((1,camCorners.shape[1]))))

# The extrinsic transformation is determined by angling (radPitch,radYaw,radRoll) and position of camera (pCamera)
T_extrinsic = extrinsicTransform(radPitch,radYaw,radRoll,pCamera)

# Image corners are converted to world coordinates. 
linePX = np.matmul(T_extrinsic,camCorners)
linePX = np.delete(linePX,3, axis=0).transpose()


# Interspection between ground plane (defined by normal and point) and the four image corner rays  defined by two points (camera position and image corner positions). 
# Define plane by point and normal vector.
pPlane = np.array([0,0,0],dtype=np.double)
nPlane = np.array([0,0,1],dtype=np.double)

# Intersection with ground plane in pDst. 
pDst,fac,val = intersection_line_plane(pCamera,linePX,pPlane,nPlane)


# Finally the image needs to be wrapped into a new image. 
pDstOut = pDst/resolution
pDstOut = pDstOut-np.min(pDstOut,axis=0)
pDstSize = np.max(pDstOut,axis=0).astype(np.int)


# The homography that maps image points to ground plane is determined. 
M = cv2.getPerspectiveTransform(pSrc[:,:2].astype(np.float32),pDstOut[:,:2].astype(np.float32))


# The image is wrapped. 
warped = cv2.warpPerspective(np.flipud(rgb[rHorizon:,:,:]), M,(pDstSize[0],pDstSize[1]) , flags=cv2.INTER_LINEAR)

print 'run time: ', (time.time()-t1)*1000,  'ms' 
#rgb = cv2.line(rgb,np.array([0, 100]),np.array([0,0]),np.array([255,255,255]),5)

# Draw horizon to image
if rHorizonTrue > 0:
    rgb[int(rHorizonTrue):int(rHorizonTrue)+3,:,:] = [255,0,0]
    rgb[int(rHorizon):int(rHorizon)+3,:,:] = [0,0,255]


plt.figure()
plt.imshow(rgb)
plt.figure()
plt.imshow(warped)


fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')
n = 100

points = np.vstack((pCamera,linePX,pDst))
X = points[:,0]
Y = points[:,1]
Z = points[:,2]
ax.scatter(X,Y,Z)
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0

mid_x = (X.max()+X.min()) * 0.5
mid_y = (Y.max()+Y.min()) * 0.5
mid_z = (Z.max()+Z.min()) * 0.5
ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
#ax.set_zlim(mid_z - max_range, mid_z + max_range)
ax.set_zlim(0, mid_z + max_range)
plt.show()