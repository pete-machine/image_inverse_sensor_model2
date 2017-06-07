
import time
import cv2
import sys
import numpy as np
import numpy.matlib as ml
from PIL import Image
from skimage.draw import polygon
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from ismFunctions import inversePerspectiveMapping, image2ogm
    
    
# Inverse Perspective Mapping
anomaly = cv2.imread('tmpImage.png',0)
theta0 = 15*np.pi/180
alpha_v = (78/2)*np.pi/180
#rHorizonRatio = 0.4630 #500/np.size(anomaly,1);
#rHorizonRatio = 0.5 #500/np.size(anomaly,1);


#rHorizon = int(np.round(np.size(anomaly,0)*rHorizonRatio)); 

grid_xSizeInM = -1.0
grid_ySizeInM = -1.0
grid_resolution = 0.01
# Get the lookup tables
(Xvis, Yvis,rHorizon) = inversePerspectiveMapping(anomaly.shape[1], anomaly.shape[0], 0, 0, 1.5, theta0 , 10*np.pi/180, alpha_v);

    
start_time = time.time()
(grid, nGridX, nGridY, dist_x, dist_y,IcroppedTransformed) = image2ogm(Xvis,Yvis,anomaly,rHorizon,grid_xSizeInM,grid_ySizeInM,grid_resolution,0.0)
print("--- image2ogm %s seconds ---" % (time.time() - start_time))
#gridOut = flipud(grid)
#b = np.ascontiguousarray(xyvalPos).view(np.dtype((np.void, xyvalPos.dtype.itemsize * xyvalPos.shape[1])))
#_, idx = np.unique(b, return_index=True)
#test = xyvalPos[idx]

imgplot1 = plt.imshow(anomaly)
#imgplot1 = plt.matshow(Icropped)
#imgplot1 = plt.matshow(gridObstacle)
#imgplot2 = plt.matshow(gridBase)
imgplot3 = plt.imshow(grid)




#plotInversePerspectiveMapping(12, uint8(anomaly), rHorizon, Xvis, Yvis, [0,0,0;1,1,1;zeros(254,3)]);