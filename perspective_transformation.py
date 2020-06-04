import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg





def unwarp(img):

	
	# Define 4 sources points : picked manually 
	src = np.float32([[620,450],[715,450],[295,680],[1088,680]])

	# Define 4 destination points
	dst = np.float32([(450,0),(img.shape[1]-450,0),(450,img.shape[0]),(img.shape[1]-450,img.shape[0])])

	M = cv2.getPerspectiveTransform(src,dst)
	warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)
	return warped



if __name__ == '__main__':
	image = mpimg.imread('test_images/test1.jpg')
	result = pipeline(image)



	#cv2.line(warped, tuple(dst[0]), tuple(dst[1]), [255,0,0], 2)
	#cv2.line(warped, tuple(dst[1]), tuple(dst[2]), [255,0,0], 2)
	#cv2.line(warped, tuple(dst[2]), tuple(dst[3]), [255,0,0], 2)
	#cv2.line(warped, tuple(dst[3]), tuple(dst[0]), [255,0,0], 2)

	# Plot the result
	f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
	f.tight_layout()

	ax1.imshow(image)
	ax1.set_title('Original Image', fontsize=40)

	ax2.imshow(result)
	ax2.set_title('Pipeline Result', fontsize=40)
	plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
	plt.show()