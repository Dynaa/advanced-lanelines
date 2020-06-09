import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg





def unwarp(img):

	
	# Define 4 sources points : picked manually 
	src = np.float32([[490,480],[800,480],[1250,720],[40,720]])

	# Define 4 destination points
	dst = np.float32([[0,0],[1280,0],[1250,720],[40,720]])

	M = cv2.getPerspectiveTransform(src,dst)
	warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)
	return warped, M



if __name__ == '__main__':
	image = mpimg.imread('test_images/test1.jpg')
	result = unwarp(image)


	# Plot the result
	f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
	f.tight_layout()

	ax1.imshow(image)
	ax1.set_title('Original Image', fontsize=40)

	ax2.imshow(result)
	ax2.set_title('Pipeline Result', fontsize=40)
	plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
	plt.show()