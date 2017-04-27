# Inverse Sensor model for camera obstacle detectors
Inverse sensor model for images. Performs inverse perspective mapping to occupancy grid maps with a specified gridmap resolution for a camera specified by its position (translation x,y,z), resolution (height and width), field of view (Diagonal FOV) and tilting (pitch, yaw).

The package supports multiple formats; 
	- Raw monochromatic image such as a grayscaled or thermal image [WxHx1].
	- Bounding boxes [x_{bb},y_{bb},w_{bb},h_{bb},prob_{bb},class_{bb}] X nBoundingBoxes
	- Segementation images [h_image X image_weidht X nClasses]

