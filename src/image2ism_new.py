
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
    
    def __init__(self,resolution=0.1,degCutBelowHorizon=10.0,minLikelihood=0.4,maxLikelihood=0.8,printProcessingTime=False):
        self.isIntrinsicUpdated = False
        self.isExtrinsicUpdated = False
        self.isHomographyUpdated = False
        self.printProcessingTime = printProcessingTime
        self.minLikelihood = minLikelihood
        self.maxLikelihood = maxLikelihood
        self.resolution = resolution
        self.degCutBelowHorizon = degCutBelowHorizon
        
    def update_homography(self,imDim_in):
        self.imDim_in = imDim_in
        if self.isIntrinsicUpdated and self.isExtrinsicUpdated:
            t0 = time.time()
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
            

            # Interspection between ground plane (defined by normal and point) and the four image corner rays  defined by two points (camera position and image corner positions). 
            # Define plane by point and normal vector.
            pPlane = np.array([0,0,0],dtype=np.double)
            nPlane = np.array([0,0,1],dtype=np.double)
            
            pRayStarts = self.pCamera # Camera position
            # Intersection with ground plane in pDst. 
            pDst,fac,val = self.intersection_line_plane(pRayStarts,pRayEnds,pPlane,nPlane)
            
            # Finally the image needs to be wrapped into a new image. 
            pDstOut = pDst/self.resolution
            pDstOut = pDstOut-np.min(pDstOut,axis=0)
            self.pDstSize = np.max(pDstOut,axis=0).astype(np.int)
            self.distToMapping = np.min(pDst[:,0])
            #rHorizon, rHorizonTrue = determineHorizon(imDim,radFOV,radPitch,degCutBelowHorizon=10)
            pSrc = np.array([[0,0],[0,self.imDim_in[0]-self.imDim_in[0]*rHorizon],[self.imDim_in[1],self.imDim_in[0]-self.imDim_in[0]*rHorizon],[self.imDim_in[1],0]])
            # The homography that maps image points to ground plane is determined. 
            self.M = cv2.getPerspectiveTransform(pSrc[:,:2].astype(np.float32),pDstOut[:,:2].astype(np.float32))   
            
            self.isHomographyUpdated = True
            if self.printProcessingTime: 
                print "Time (Homography): ", (time.time()-t0)*1000, "ms"
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
        
        radCutBelowHorizon = degCutBelowHorizon*np.pi/180.0
        
        # Estimate horison only from pitch        
        rHorizonTrue = (self.radFOV[0]/2-self.radPitch)/self.radFOV[0]
        rHorizon = rHorizonTrue+radCutBelowHorizon/self.radFOV[0]
    
        
        #print 'rHorizonTrue: ', rHorizonTrue, 'rHorizon: ', rHorizon
        
        rHorizon = np.maximum(rHorizon,0)
        rHorizonTrue = np.maximum(rHorizonTrue,0)
        
        return  rHorizon, rHorizonTrue
    
    def update_intrinsic_from_CameraInfo(self,CameraInfo):
        self.imDimOrg = [CameraInfo.height,CameraInfo.width];
        self.K = np.reshape(CameraInfo.K,(3,3))
        self.Kinv = np.linalg.inv(self.K)
        focal_length = np.array([self.K[0,0],self.K[1,1]])
        self.radFOV = 2*np.arctan2(np.array(self.imDimOrg),2*focal_length)
        self.isIntrinsicUpdated = True
        
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
        t0 = time.time()
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
        if self.printProcessingTime: 
                print "Time (Extrinsic): ", (time.time()-t0)*1000, "ms"
        return T
    
    def convert2grid(self,imgIn,minLikelihood,maxLikelihood):
         # Expect 0-255 and convert to grid format that ranges between 0-100...
         # However, 0 mappes to minLikelihood(0.4), 0> to 255 mappes between minLikelihood(0.5) to maxLikelihood (0.8)
        imgIn = imgIn.astype(np.float32)
        mask = imgIn>0
        imgIn[mask] = imgIn[mask]*(maxLikelihood-0.5)/255.0+0.5
        imgIn[mask==False] = minLikelihood
        imgIn = (imgIn*100).astype(np.uint8)        
        return imgIn
        
    def makePerspectiveMapping(self,imgIn,match2Grid = False ):

        if self.isHomographyUpdated: 
            t0 = time.time()
            if match2Grid: 
                imgIn = self.convert2grid(imgIn,self.minLikelihood,self.maxLikelihood)
                borderValue=50
            else:
                # Invert and normalize output between 0-100
                imgIn_f = (imgIn-np.min(imgIn)).astype(np.float32)
                imgIn = (100*(1-imgIn_f/np.max(imgIn_f))).astype(np.uint8)
                
                borderValue=0
                
            out = cv2.warpPerspective(np.flipud(imgIn), self.M,(self.pDstSize[0],self.pDstSize[1]) , flags=cv2.INTER_NEAREST,borderValue=borderValue)
            if self.printProcessingTime: 
                print "Time (Perspective Mapping): ", (time.time()-t0)*1000, "ms"
            return out
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
            outStr = outStr + "    Original Image size " + str(self.imDimOrg)  + "\n"
            outStr = outStr + "    FOV_ver/hor (Radians) " + str(self.radFOV)  + "\n"
            outStr = outStr + "    FOV_ver/hor (Degrees) " + str(self.radFOV*180/np.pi) + "\n"
            outStr = outStr + "    K: " + str(self.K).replace('\n','\n       ') + "\n"
            
        if self.isExtrinsicUpdated:
            outStr = outStr + "Extrinsic: \n    Pitch (radian): " + str(self.radPitch) + " \n    Yaw (radian): " + str(self.radYaw) + " \n    Roll (radian): " + str(self.radRoll) + "\n"
            outStr = outStr + "    Camera Position (m): " + str(self.pCamera) + "\n"
            outStr = outStr + "    T: " + str(self.T_extrinsic).replace('\n','\n       ') + "\n"
            
         
        if self.isHomographyUpdated:
            outStr = outStr + "Homography:\n    M: " + str(self.M).replace('\n','\n       ') + "\n"
            outStr = outStr + "    Orig. Image size " + str(self.imDimOrg)  + "\n"
            outStr = outStr + "    Input image size " + str(np.array(self.imDim_in))  + "\n"
        
        outStr = outStr + "################################################ \n"
        return outStr