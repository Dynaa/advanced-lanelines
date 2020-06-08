import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import argparse
import glob
from camera_calibration import undistort
from perspective_transformation import unwarp
from image_processing import color_thresolding
from finding_lane import fit_polynomial
from finding_lane import search_around_poly






""" Test overall image pipeline line, from undistort to line and curvature computation """

if __name__ == '__main__':
	image = mpimg.imread('test_images/test1.jpg')

	# Apply calibration parameters to undistort image
	undistort_image = undistort(image)


	# Unwarp image 
	unwarp = unwarp(image)
	cv2.imwrite('output_images/unwarp_straight_lines2.jpg', cv2.cvtColor(unwarp, cv2.COLOR_RGB2BGR))

	# Apply thresold 
	color = color_thresolding(unwarp, 0)
	cv2.imwrite('output_images/image_processing_output/thresold_straight_lines2.jpg', color)
	ploty = np.linspace(0, color.shape[0]-1, color.shape[0] )
	#left_fitx, right_fitx, polynomial = fit_polynomial(color)
	polynomial = search_around_poly(color)

	x0, y0 = 490,480
	x1, y1 = 800,480
	x2, y2 = 1250,720
	x3, y3 = 40,720

	cv2.circle(image, (x1, y1), 2, (255, 0, 0), 2) 
	cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
	cv2.line(image, (x0,y0), (x1,y1), [255,0,0], 2)
	cv2.line(image, (x1,y1), (x2,y2), [255,0,0], 2)
	cv2.line(image, (x2,y2), (x3,y3), [255,0,0], 2)
	cv2.line(image, (x3,y3), (x0,y0), [255,0,0], 2)


	# Plot the result
	f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
	f.tight_layout()

	ax1.imshow(image)
	ax1.set_title('Original Image', fontsize=40)

	ax2.imshow(polynomial)
	ax2.set_title('Pipeline Result', fontsize=40)
	#ax2.plot(left_fitx, ploty, color='yellow')
	#ax2.plot(right_fitx, ploty, color='yellow')
	# Draw the lane onto the warped blank image
	cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
	cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
	result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

	# Plot the polynomial lines onto the image
	plt.plot(left_fitx, ploty, color='yellow')
	plt.plot(right_fitx, ploty, color='yellow')
	plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
	plt.show()