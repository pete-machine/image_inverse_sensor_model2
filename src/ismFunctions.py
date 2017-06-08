
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

tStart = time.time()
def inversePerspectiveMapping(width, height, x, y, z, pitch, yaw, alpha):

#    % Returns a xy-lookup table in the base coordinate system
#    %
#    % Note: "roll" (s.t. rotation around x-axis is not implemented)
#    % Input:
#    %  width: Width of the original image (pixel)
#    %  height: height of the original image (pixel)
#    %  horizon: Horizon in pixel to cut the image
#    %  x: Tranlation in x direction in the planar camera frame (meter)
#    %  y: Tranlation in y direction in the planar camera frame (meter)
#    %  z: Tranlation (elevation) in z direction of the camera (meter)
#    %  pitch: Rotation around y-axis (0 = camera facing stright driving orientation, <0 = camera facing upward, >0 camera facing downward) (rad)
#    %  yaw: Rotation around z-axis (rad)
#    %  alpha: Half total viewing angle from corner to corner (rad)
#    % 
#    % Output:
#    %  Xvis: Lookup table for each pixel as x-values in the origin coordinate system (meter)
#    %  Yvis: Lookup table for each pixel as y-values in the origin coordinate system (meter)

    #addpath(genpath('../functions'))

    # Now grab the image size.
    m = np.array(height)
    n = np.array(width)

    # The row where the horizon would appear if it weren't for the mountains
    # (if we were really on a planar road) seems to be about:
    #rHorizon = horizon;
    rHorizon = int(np.ceil( (m-1)/2*(1 - np.tan(pitch)/np.tan(alpha)) + 1 ));
    rHorizon = rHorizon + int(m*0.05); # To be sure


    h = z; # Height over feature
    transM = np.array([[1, 0, 0, x],[0, 1, 0, y],[0, 0, 1, 0],[0, 0, 0, 1]])
    rotM = np.array([[np.cos(yaw), -np.sin(yaw), 0, 0],[np.sin(yaw), np.cos(yaw), 0, 0],[0, 0, 1, 0],[0, 0, 0, 1]])
    #T = trans(x,y,0)*rot(yaw,'z'); # Planar tranformtion from projected camera frame to vehicle frame
    T = transM.dot(rotM) # Planar tranformtion from projected camera frame to vehicle frame
    
    # To make sure the relationships between alpha_u and alpha_v are correct,
    # let's just drive both by setting alpha_tot and calculating them.
    # Assume a 30 degree viewing angle for now since that seems pretty
    # reasonable.
    alpha_tot = alpha;

    # Get the corresponding vertical and horizontal viewing angles.
    den = np.sqrt(np.power(m-1,2)+np.power(n-1,2));
    alpha_u = np.arctan( (n-1)/den * np.tan(alpha_tot) );
    alpha_v = np.arctan( (m-1)/den * np.tan(alpha_tot) );
    # The guesses for alpha_tot and rH give a camera declination angle of roughly:
    theta0 = pitch;

    
    ## IPM: Get Xvis Yvis Matrices

    # If we take from that row down in the cropped image, that leaves us with:
    mCropped = m-rHorizon+1 # rows in the cropped image.

#    Xvis = zeros(mCropped,n);
#    Yvis = zeros(size(Xvis));
    Xvis = np.zeros((mCropped,n),np.uint8)
    Yvis = np.zeros(Xvis.shape, np.uint8)

    # PART 1c: optimized twice (0.002775s)
    r = range(0,mCropped)
    rOrig = r + np.array(rHorizon)
    rFactor = (1.0-2.0*(rOrig-1.0)/(float(m)-1.0))*np.tan(alpha_v)
    num = 1.0 + rFactor*np.tan(theta0);
    den = np.tan(theta0) - rFactor;
    #Xvis = ml.repmat(h*(num./den))',[1 n]);
    Xvis = ml.repmat((h*(num/den)), n,1).T
    
    c = np.array(range(0,n))
    num = (1.0-2.0*c/(float(n)-1.0))*np.tan(alpha_u);
    den = np.sin(theta0) - rFactor*np.cos(theta0);
    
    Yvis = h*(num/np.reshape(den,(-1,1)));
    
    
    
    
    tmp = T.dot(np.column_stack((Xvis.flatten(),Yvis.flatten(),np.zeros((Yvis.size,)),np.ones((Yvis.size,)))).T)
    Xvis = tmp[0,].reshape(np.shape(Xvis))
    Yvis = tmp[1,].reshape(np.shape(Xvis))
    return (Xvis, Yvis,rHorizon)

# Xvis,Yvix: Defines lookup table from above
# inputImage: A monochromatic "image" in [0-255] remapped in the range [0.5-0.8]. 
# 		Areas: 
#			visibleArea: In visible areas, (and if nothing is detected) the likelihood of an obstacle is less than 0.5. (E.g. 0.4) 
#			pNonVisible: In non-visible areas (outside the camera field-of-view), the likelihood of an obstacle is typically 0.5.
#			pMaxVisible: In detected areas, the likelihood of an obstacle is between 0.5 and the maximum likelihodd (e.g. 0.8). The input image in range [0;255] is mapped between 0.5 to 0.8(pMaxVisible). 

def image2ogm(Xvis,Yvis,inputImage,rHorizon,grid_xSizeInM,grid_ySizeInM,grid_resolution,objectExtent,minLikelihood,maxLikelihood):
        
    mmX = np.array([Xvis.min(), Xvis.max()]); 
    mmY = np.array([Yvis.min(), Yvis.max()]); 
    diffX = np.diff(mmX);
    diffY = np.diff(mmY);

    resolution = grid_resolution #m/cell
    
    if ((grid_xSizeInM<0) | (grid_ySizeInM<0)): 
        # Grid size in meter (scales by default to the visible area)
        gridSizeXIn = np.ceil(diffX/resolution)*resolution; # In meter
        gridSizeYIn = np.ceil(diffY/resolution)*resolution; # In meter
    else :
        # Grid size in meter
        gridSizeXIn = grid_xSizeInM
        gridSizeYIn = grid_ySizeInM
        
    nGridX = int(gridSizeXIn/resolution);
    nGridY = int(gridSizeYIn/resolution);  
    Icropped = inputImage[rHorizon-2:-1,:]

#    try:
    IcroppedTransformed = TransformImageAccordingToObjectExtend(Xvis,Yvis,Icropped,objectExtent)


#    except: 
#        print "ERROR: Save mage: shape: ", inputImage.shape 
#        cv2.imwrite('/home/repete/CRAP/testImage' + str(time.time()-tStart) + '.png',Icropped)
    dist_x = mmX[0]
    dist_y = mmY[0]
    
    Xtrans = np.array((Xvis-mmX[0])/resolution);
    Ytrans = np.array((Yvis-mmY[0])/resolution);
    
    
    
    
    pNonVisibleAreaFloat = 0.5;
    pNonVisibleArea = 255*pNonVisibleAreaFloat
    pVisibleAreaFloat = minLikelihood; # Originally set to 0.4
    pVisibleArea = 255*pVisibleAreaFloat
    pMaxLikelihoodFloat = maxLikelihood; # Originally set to 0.8
    pMaxLikelihood = 255*pMaxLikelihoodFloat
    factor = (pMaxLikelihoodFloat-pNonVisibleAreaFloat);
    # maxVisualDistance = 8; % in Meter
    # maxVisualGridDistance = (maxVisualDistance-mmX(1))/resolution;

    
    ## gridBase2 = zeros(nGridY,nGridX);    
    
    gridBase = pNonVisibleArea*np.ones((nGridY,nGridX))
    mask = np.zeros((nGridY,nGridX))
    ptFovX = np.array([Xtrans[0,0], Xtrans[-1,0], Xtrans[-1,-1], Xtrans[0,-1], Xtrans[0,0]]);
    ptFovY = np.array([Ytrans[0,0], Ytrans[-1,0], Ytrans[-1,-1], Ytrans[0,-1], Ytrans[0,0]]);
    rr, cc = polygon(ptFovY, ptFovX)
    mask[rr,cc] = pVisibleArea-pNonVisibleArea
    gridBase = mask+gridBase;
    
    
    
    gridObstacle = np.zeros((nGridY,nGridX))
    
    
    linIndex = np.squeeze(IcroppedTransformed>0.);
    yindex = Ytrans[linIndex].astype(int)
    xindex = Xtrans[linIndex].astype(int)
    gridObstacle[yindex,xindex] = pNonVisibleArea+IcroppedTransformed[linIndex]*factor;
    gridBase[yindex,xindex] = 0;
    
    # Negative values are set to 0.5. (This option is used ignore areas in the image). 
    linIndex = np.squeeze(IcroppedTransformed<-1.);
    if linIndex.size > 0:
        yindex = Ytrans[linIndex].astype(int)
        xindex = Xtrans[linIndex].astype(int)
        gridObstacle[yindex,xindex] = pNonVisibleArea;
        gridBase[yindex,xindex] = 0;

    grid = ((gridBase+gridObstacle)*100)/255
    grid = np.uint8(grid)
    return (grid, nGridX, nGridY, dist_x, dist_y,IcroppedTransformed)
    
def TransformImageAccordingToObjectExtend(Xvis,Yvis,inputImageCropped,objectExtent):
    # In regular image domain, the first object in each column is extended to fill up (in the image)
    # a specified space in the inverse sensor model. Areas behind the object is set to 0.5.
    # Object areas is filled with average probability of first object. 
    
    # The output image is only changed if the object extent is bigger than 0.
    if(objectExtent>0):
        newMap = np.zeros(inputImageCropped.shape)
        newMap[inputImageCropped<0] = -10
        # 
        for iCol in range(0,inputImageCropped.shape[1]): #range(450,451):#
            #try:
            sectionTmp = np.squeeze(np.nonzero(inputImageCropped[:,iCol]>0))
            if sectionTmp.shape:
                sectionCol = np.flipud(sectionTmp)
                
                if sectionCol.shape[0]>0:
                    
                    # Returns the the pixel, when the first component (from the bottom) in the image stops or a gap appears.
                    startPixel = np.squeeze(sectionCol[np.squeeze(np.where(np.diff(sectionCol)<-1))])
        
                    if startPixel.shape:
                        if startPixel.shape[0] > 0:
                            startPixel = startPixel[0]
                        else:
                            startPixel = sectionCol[-1]

                    # Returns the 
                    endPixel = sectionCol[0]
                    probability = np.mean(inputImageCropped[startPixel:endPixel,iCol])
                    #endPixelXvis = np.min(endPixel,Xvis.shape[0])
                    distanceFromStartOfObject = Xvis[0:endPixel,iCol]-Xvis[endPixel-1,iCol]

                    # Finds the extent of an object in the image based on the objectExtent in the inverse sensor model.
                    if distanceFromStartOfObject.shape[0]>1:
                        endPixelOfISM = np.squeeze(np.nonzero(distanceFromStartOfObject<objectExtent))[0]
                        newMap[endPixelOfISM:endPixel,iCol] = probability
                        newMap[0:endPixelOfISM,iCol] = 1                        
                    
    else:
        # Nothing is done if the object has an extent of 0
        newMap = inputImageCropped
    return newMap
    

## Inverse Perspective Mapping
#anomaly = 255*(cv2.imread('tmpImage2.png',0)/255)
#rHorizon = 500 # nedds to be under the horizon for the given pitch
#grid_xSizeInM = -1.0
#grid_ySizeInM = -1.0
#grid_resolution = 0.1
#objectExtend = 10.0
## Get the lookup tables
#
#(Xvis, Yvis,rHorizon) = inversePerspectiveMapping(anomaly.shape[1], anomaly.shape[0], 0, 0, 1.5, 20*np.pi/180, 10*np.pi/180, 20*np.pi/180);
#
#    
#start_time = time.time()
#grid,nGridX,nGridY, dist_x1, dist_y1,IcroppedTransformed = image2ogm(Xvis,Yvis,anomaly,rHorizon,grid_xSizeInM,grid_ySizeInM,grid_resolution,objectExtend)
#print("--- image2ogm %s seconds ---" % (time.time() - start_time))
##gridOut = flipud(grid)
#
#
##imgplot1 = plt.matshow(linIndex)
##imgplot1 = plt.matshow(Icropped)
#imgplot1 = plt.matshow(anomaly)
#imgplot2 = plt.matshow(IcroppedTransformed)
#imgplot3 = plt.matshow(grid)


