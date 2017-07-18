
import time
import sys
import numpy as np

import sys
sys.path.append("/usr/lib/python2.7/dist-packages")
import cv2

from skimage.draw import polygon
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#rHorizon = 0
class InversePerspectiveMapping:
    
    def __init__(self,resolution=0.1,degCutBelowHorizon=10.0):
        self.isIntrinsicUpdated = False
        self.isExtrinsicUpdated = False
        self.isHomographyUpdated = False
        self.resolution = resolution
        self.degCutBelowHorizon = degCutBelowHorizon
        
    def update_homography(self,imDim):
        print "H5_0"
        if self.isIntrinsicUpdated and self.isExtrinsicUpdated:
            print  "H5_1"
            rHorizon, rHorizonTrue = self.determineHorizon(degCutBelowHorizon=self.degCutBelowHorizon)
                
            # Image corners in matrix. 
            imCorners = np.array([[0,0],[0,self.imDimOrg[0]-self.imDimOrg[0]*rHorizon],[self.imDimOrg[1],self.imDimOrg[0]-self.imDimOrg[0]*rHorizon],[self.imDimOrg[1],0]])
            imCorners = np.hstack((imCorners,np.ones((imCorners.shape[0],1))))
            
            # image corners are converted to image space. 
            camCorners = np.matmul(self.Kinv,imCorners.T)
            
            # Flip axis and make coordinates 3D homogeneous. 
            camCorners = np.vstack((camCorners[2,:],camCorners[0,:],camCorners[1,:],np.ones((1,camCorners.shape[1]))))
            
            
            
            # Image corners are converted to world coordinates. 
            pRayEnds = np.matmul(self.T_extrinsic,camCorners)
            pRayEnds = np.delete(pRayEnds,3, axis=0).transpose()
            
            print  "H5_2"
            # Interspection between ground plane (defined by normal and point) and the four image corner rays  defined by two points (camera position and image corner positions). 
            # Define plane by point and normal vector.
            pPlane = np.array([0,0,0],dtype=np.double)
            nPlane = np.array([0,0,1],dtype=np.double)
            
            pRayStarts = self.pCamera # Camera position
            # Intersection with ground plane in pDst. 
            pDst,fac,val = self.intersection_line_plane(pRayStarts,pRayEnds,pPlane,nPlane)
            
            print  "H5_3"
            # Finally the image needs to be wrapped into a new image. 
            pDstOut = pDst/self.resolution
            pDstOut = pDstOut-np.min(pDstOut,axis=0)
            self.pDstSize = np.max(pDstOut,axis=0).astype(np.int)
            
            #rHorizon, rHorizonTrue = determineHorizon(imDim,radFOV,radPitch,degCutBelowHorizon=10)
            pSrc = np.array([[0,0],[0,imDim[0]-imDim[0]*rHorizon],[imDim[1],imDim[0]-imDim[0]*rHorizon],[imDim[1],0]])
            # The homography that maps image points to ground plane is determined. 
            self.M = cv2.getPerspectiveTransform(pSrc[:,:2].astype(np.float32),pDstOut[:,:2].astype(np.float32))   
            
            self.isHomographyUpdated = True
            print  "H5_4"
            return pRayEnds,pDst,rHorizon, rHorizonTrue,pSrc,pDstOut
        else:
            raise NameError('update_homography requires both InversePerspectiveMapping.update_intrinsic() and InversePerspectiveMapping.update_extrinsic() functions to be executed. ')
            return []
        
    # intersection function
    def intersection_line_plane(self,p0, p1, p_co, p_no, epsilon=1e-6):
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
    
    def determineHorizon(self, degCutBelowHorizon=10.0):
        # Estimate horison only from pitch
        radCutBelowHorizon = degCutBelowHorizon*np.pi/180.0
        #cutBelowHorizon = degCutBelowHorizon*np.pi/180 # 10 degrees
        rHorizonTrue = (self.radFOV[0]/2-self.radPitch)/self.radFOV[0]
        #rHorizon = (radFOV[0]/2-radPitch-cutBelowHorizon)/radFOV[0]
        rHorizon = rHorizonTrue+radCutBelowHorizon/self.radFOV[0]
        #(radFOV[0]/2+cutBelowHorizon-radPitch)
    #    rHorizon =  int(np.ceil( (imDim[0]-1)/2*(1 - np.tan(radPitch-cutBelowHorizon)/np.tan(radFOV[0]/2)) + 1 ));
    #    rHorizonTrue = int(np.ceil( (imDim[0]-1)/2*(1 - np.tan(radPitch)/np.tan(radFOV[0]/2)) + 1 ));
    #    rHorizon =  int(np.ceil( imDim[0]/2*(1 - np.tan(radPitch-cutBelowHorizon)/np.tan(radFOV[0]/2))));
    #    rHorizonTrue = int(np.ceil( imDim[0]/2*(1 - np.tan(radPitch)/np.tan(radFOV[0]/2))));
    #    rHorizon =  1.0/2*(1 - np.tan(radPitch-cutBelowHorizon)/np.tan(radFOV[0]/2));
    #    rHorizonTrue = 1.0/2*(1 - np.tan(radPitch)/np.tan(radFOV[0]/2));
        
        print 'rHorizonTrue: ', rHorizonTrue, 'rHorizon: ', rHorizon
        
        rHorizon = np.maximum(rHorizon,0)
        rHorizonTrue = np.maximum(rHorizonTrue,0)
        
        return  rHorizon, rHorizonTrue
    
    def update_intrinsic(self,focal_length, principal_point_offset,imDimOrg,skew=0):
        
        self.imDimOrg = imDimOrg
        
        ### INIT #############
        # DETERMINE K 
        # Camera intrinsic matrix is determined. 
        K = np.eye(3)
        K[[0,1],[0,1]] = focal_length; 
        K[0:2,2] = principal_point_offset;
        K[0,1] = skew
        #K = np.array([[fl[0], s, ppo[0]],[0,fl[1],ppo[1]],[0,0,1]])
        self.K = K
        self.Kinv = np.linalg.inv(K)
        # 2*atan2(imageSize./2,focalLength);
#        self.radFOV = 2*np.arctan(imDimOrg/2,focal_length)

        self.radFOV = 2*np.arctan2(np.array(imDimOrg),2*focal_length)
        self.isIntrinsicUpdated = True
        
        
    # Angling of camera: pitch, yaw, roll of
    # Camera position: pCam = [x,y,z] = [x,y,height]
    def update_extrinsic(self,pitch,yaw,roll, pCamera, inDeg = False):
        
        # In degree convert to radians
        if inDeg:
            pitch = pitch*np.pi/180
            yaw = yaw*np.pi/180
            roll = roll*np.pi/180
        
        T_pitch = np.array([[np.cos(pitch),0,np.sin(pitch),0],[0,1,0,0],[-np.sin(pitch),0,np.cos(pitch),0],[0,0,0,1]])
        T_yaw = np.array([[np.cos(yaw), -np.sin(yaw),0,0],[np.sin(yaw),np.cos(yaw),0,0],[0,0,1,0],[0,0,0,1]])
        T_roll = np.array([[1,0,0,0],[0,np.cos(roll),-np.sin(roll),0],[0,np.sin(roll),np.cos(roll),0],[0,0,0,1]])
    
        
        self.radPitch = pitch
        self.radYaw = yaw
        self.radRoll = roll
        
        T = np.matmul(T_roll,np.matmul(T_pitch,T_yaw))
        T[0:3,3] = pCamera 
        
        self.T_extrinsic = T
        self.pCamera = pCamera
        self.isExtrinsicUpdated = True
        
        return T
    
    def makePerspectiveMapping(self,imgIn):
        print  "H8_4"
        if self.isHomographyUpdated: 
            ### Make warping ########
#            print "M: ", self.M
#            print "imgIn: ", imgIn.shape
#            print "pDstSize: ", self.pDstSize
            return cv2.warpPerspective(np.flipud(imgIn), self.M,(self.pDstSize[0],self.pDstSize[1]) , flags=cv2.INTER_LINEAR)
        else:
            raise NameError('makePerspectiveMapping requires the InversePerspectiveMapping.update_homography-function to be executed')
    def __repr__(self):
        return "InversePerspectiveMapping()"
    
    def strYes(self, boolTrue): 
        if boolTrue:
            return "(Yes)"
        else:
            return "(No)"
    def __str__(self):
        outStr = "\n######## INVERSE PERSPECTIVE MAPPING ###########\n"
        outStr = outStr + "Resolution: " + str(self.resolution) + ", Cut horizon by: " + str(self.degCutBelowHorizon) + "degrees \n"
        outStr = outStr + "Updated: Intrisic " +  self.strYes(self.isIntrinsicUpdated) + " Extrinsic " + self.strYes(self.isExtrinsicUpdated) + " Homography " + self.strYes(self.isHomographyUpdated) + "\n"
        
        if self.isIntrinsicUpdated: 
            outStr = outStr + "Intrinsic: \n"
            outStr = outStr + "    FOV_ver/hor (Radians) " + str(self.radFOV)  + "\n"
            outStr = outStr + "    FOV_ver/hor (Degrees) " + str(self.radFOV*180/np.pi) + "\n"
            outStr = outStr + "    K: " + str(self.K).replace('\n','\n       ') + "\n"
            
        if self.isExtrinsicUpdated:
            outStr = outStr + "Extrinsic: \n    Pitch (radian): " + str(self.radPitch) + " \n    Yaw (radian): " + str(self.radYaw) + " \n    Roll (radian): " + str(self.radRoll) + "\n"
            outStr = outStr + "    Camera Position (m): " + str(self.pCamera) + "\n"
         
        if self.isHomographyUpdated:
            outStr = outStr + "Homography:\n    M: " + str(self.M).replace('\n','\n       ') + "\n"
        
        outStr = outStr + "################################################ \n"
        return outStr