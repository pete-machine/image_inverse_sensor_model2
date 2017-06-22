#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 09:36:31 2017

@author: pistol
"""
import numpy as np
from scipy.stats import multivariate_normal
from scipy.spatial.distance import cdist
from skimage.draw import polygon


class Bb2ism:

    def __init__(self, hFOV, localizationErrorStd,localizationErrorStdEnd = [], degradOutlook = False, degradOutlookAfterM=0.0, max_distance=50,grid_resolution=0.1,pVisible=0.4 ,pNonVisible=0.5,pMaxLikelyhood=0.8):

        
        self.grid_resolution = grid_resolution
        self.pVisible = pVisible
        self.pNonVisible = pNonVisible
        self.pMaxLikelyhood = pMaxLikelyhood
        
        # Base map grid
        gridHeightDistance = 2*(np.tan(hFOV/2)*max_distance)
        gridWidthDistance = max_distance
        self.gridHeight = int(round(gridHeightDistance/self.grid_resolution))
        self.gridWidth = int(round(gridWidthDistance/self.grid_resolution))
        
        # Empty grid map
        self.baseGrid = pNonVisible*np.ones((self.gridHeight,self.gridWidth))
        self.gridOriginXYZ = np.array([0,-gridHeightDistance/2.0,0])
        #detectionGrid = np.zeros((self.gridHeight,self.gridWidth))
        
        
        if(localizationErrorStdEnd == []):
            localizationErrorStdEnd = np.copy(localizationErrorStd)
        # Gaussian mask grid
        scaleStd = localizationErrorStd/self.grid_resolution
        scaleStdEnd = localizationErrorStdEnd/self.grid_resolution
        #print("localizationErrorStd: ",localizationErrorStd,localizationErrorStd/self.grid_resolution,"localizationErrorStdEnd: ",localizationErrorStdEnd,localizationErrorStdEnd/self.grid_resolution)
        #print("scaleStd: ",scaleStd,"scaleStdEnd: ",scaleStdEnd)
        X = np.array([0.0,max_distance/self.grid_resolution])
        self.aStd = np.array([np.diff([scaleStd[0],scaleStdEnd[0]])/np.diff(X),np.diff([scaleStd[1],scaleStdEnd[1]])/np.diff(X)])
        self.bStd = np.expand_dims(scaleStd,1)
        
        
        rowShift = self.gridHeight/2
        colShift = 0
        self.coord2mapTranslation = np.array([rowShift,colShift])
        self.coord2mapRotation = 0.0
        
        degradOutlookAfter = degradOutlookAfterM/self.grid_resolution
        
        # Visible area
        if(pVisible != pNonVisible):
            ptFovY = np.array([0.5,1.0,0.0])*self.gridHeight
            ptFovX = np.array([0.0,1.0,1.0])*self.gridWidth
            rr, cc = polygon(ptFovY, ptFovX)
            
            # 
            if(degradOutlook):
                # Degrading certainty of how well 
                if(True):
                    rrcc = np.vstack((rr,cc)).T
                    ptBase = np.array(self.coord2mapTranslation,ndmin=2);
                    baseValues = np.maximum(cdist(ptBase,rrcc)-degradOutlookAfter,0)
                    baseValues = baseValues*(pNonVisible-pVisible)/np.max(baseValues)
                
                else:
                    baseValues = np.maximum(cc.astype(float)-degradOutlookAfter,0)
                        
                baseValues = baseValues*(pNonVisible-pVisible)/np.max(baseValues)
                self.baseGrid[rr,cc] = pVisible+baseValues
            else:
                self.baseGrid[rr,cc] = pVisible
        

    def drawDetection(self, xy,detectionGrid):
        xy = np.array(xy)
        ptAngle = -np.arctan2(xy[1],xy[0])
        
            
        # If you wanna use matrix-multiplications for transforming points
        #MrotateTranslate = np.array([[np.cos(yaw), -np.sin(yaw), rowShift],[np.sin(yaw), np.cos(yaw), colShift],[0, 0, 1]])
        #Mscale = np.array([[grid_resolution,0,0],[0,grid_resolution,0],[0,0,1]])
        # Convert point to grid point
        rc_grid = (np.array([xy[1],xy[0]])/self.grid_resolution+self.coord2mapTranslation).astype(int)
        
        rcDist = np.sqrt(np.sum(xy**2))/self.grid_resolution;
        # Rotation and scaling. 
        sigma = np.eye(2)
        Mrot = np.array([[np.cos(ptAngle), -np.sin(ptAngle),], [np.sin(ptAngle),np.cos(ptAngle)]]);    
        
        useSingleTransform = False;
        
        # Calculate new localization error (proportional to distance)
        std = self.aStd*rcDist+self.bStd
        
        
        # Gassian grid mask (determined for each detection)
        maskSizeMVGInSTD = 2.5
        maxStd = np.max(std)
        maxSizeMVG = np.round(maxStd*2*maskSizeMVGInSTD/2)*2+1
        ptCenterTmp = (maxSizeMVG-1)/2
        self.ptCenter = np.array([ptCenterTmp,ptCenterTmp])
        
        # Create grid
        x, y = np.mgrid[0:maxSizeMVG, 0:maxSizeMVG]
        self.pos = np.empty(x.shape + (2,))
        self.pos[:, :, 0] = x 
        self.pos[:, :, 1] = y
        
        #std = [self.scaleStd[0],self.scaleStd[1]]
        if(useSingleTransform):
            # Rotate and scale the identity matrix (eye)
            Mscale = np.array([[std[0],0], [0,std[1]]]);
            M = Mrot.dot(Mscale)
        else:
            sigma[0,0] = std[0]**2
            sigma[1,1] = std[1]**2
            M = Mrot
        sigma = M.dot(sigma.dot(M.T))
            
        # Create normal distribution. (This is done for each point)
        rv = multivariate_normal(self.ptCenter, sigma)
        z = rv.pdf(self.pos)
        
        # Scale gaussian map. 
        #mvgRange = (pMaxLikelyhood-pVisible)
        
        #z[z>0.10] = z[z>0.10]
        dimMask = z.shape
        # Indices for cropping section from big map (same size as the gaussian mask)
        rCrop = np.array([rc_grid[0]-dimMask[0]/2,rc_grid[0]+dimMask[0]/2+1])
        cCrop = np.array([rc_grid[1]-dimMask[1]/2,rc_grid[1]+dimMask[0]/2+1])
        
        # Indices for cropping gaussian mask (The lines are only added for border cases).
        rCropZ = np.array([-1*np.minimum(0,np.min(rCrop)),dimMask[0]+np.minimum(0,self.gridHeight-np.max(rCrop))])
        cCropZ = np.array([-1*np.minimum(0,np.min(cCrop)),dimMask[1]+np.minimum(0,self.gridWidth -np.max(cCrop))])
        
        # Indices for only cropping a valid mask. Indecies must be bigger than 0 and bigger than image width and height. 
        rCrop = np.minimum(np.maximum(rCrop,0),self.gridHeight)
        cCrop = np.minimum(np.maximum(cCrop,0),self.gridWidth)
        
        # Two methods for placing a guassian distribution at the center of a detection in the map (Use method 2)
        method1 = False
        if(method1):
            mvgRange = (self.pMaxLikelyhood-self.pVisible)
            
            # Scale gaussian distribution.
            z = z*mvgRange/np.max(z)
            detectionGrid[rCrop[0]:rCrop[1],cCrop[0]:cCrop[1]] = detectionGrid[rCrop[0]:rCrop[1],cCrop[0]:cCrop[1]]+z[rCropZ[0]:rCropZ[1],cCropZ[0]:cCropZ[1]]
            
        else:
            mvgRange = (self.pMaxLikelyhood-self.pNonVisible)
            
            # Normalize distribution.
            z = z*mvgRange/np.max(z)
            
            # Many weired tricks to place the gaussian in the map. 
            edgeMax = np.max([np.max(z[0,:]),np.max(z[:,0])])
            cropZ = z[rCropZ[0]:rCropZ[1],cCropZ[0]:cCropZ[1]]
            cropBase = detectionGrid[rCrop[0]:rCrop[1],cCrop[0]:cCrop[1]]
            cropBase[cropZ>edgeMax] = cropZ[cropZ>edgeMax]+self.pNonVisible+np.maximum(0,cropBase[cropZ>edgeMax]-0.5)
            detectionGrid[rCrop[0]:rCrop[1],cCrop[0]:cCrop[1]] = cropBase
            
        return detectionGrid
    
    def drawDetections(self, xyz):
        detectionGrid = np.copy(self.baseGrid)
        print("xyz: ", xyz)
        self.nDetections = len(xyz)
        
        # Draw in all detections
        for iPoint in range(self.nDetections):
            detectionGrid = self.drawDetection(xyz[iPoint],detectionGrid)
        
        # Scale from [0.0;1.0] to [0;100].
        return detectionGrid