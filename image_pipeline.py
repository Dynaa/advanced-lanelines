import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import argparse
import glob
from matplotlib import gridspec
from camera_calibration import undistort
from perspective_transformation import unwarp
from image_processing import color_thresolding
from finding_lane import fit_polynomial
from finding_lane import search_around_poly
from finding_lane import measure_curvature_real
from finding_lane import *
from moviepy.editor import VideoFileClip
from Line import Line



left_line = Line()
right_line = Line()

def display_lane_unwarp(left_lane_inds, right_lane_inds, left_fitx, right_fitx, ploty, color):
	margin = 100
	out_img = np.dstack((color, color, color))*255
	window_img = np.zeros_like(out_img)
	nonzero = color.nonzero()
	nonzeroy = np.array(nonzero[0])
	nonzerox = np.array(nonzero[1])
	# Color in left and right line pixels
	out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
	out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

	# Generate a polygon to illustrate the search window area
	# And recast the x and y points into usable format for cv2.fillPoly()
	left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
	left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, 
	                          ploty])))])
	left_line_pts = np.hstack((left_line_window1, left_line_window2))
	right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
	right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, 
	                          ploty])))])
	right_line_pts = np.hstack((right_line_window1, right_line_window2))

	# Draw the lane onto the warped blank image
	cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
	cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
	result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
	cv2.imwrite('output_images/lane_finding_output/lane_test6.jpg', out_img)


	# Plot the result
	f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
	f.tight_layout()

	ax1.imshow(image)
	ax1.set_title('Original Image', fontsize=40)

	ax2.imshow(result)
	ax2.set_title('Pipeline Result', fontsize=40)
	# Plot the polynomial lines onto the image
	ax2.plot(left_fitx, ploty, color='yellow')
	ax2.plot(right_fitx, ploty, color='yellow')
	plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
	plt.show()

def image_pipeline(file_name, show=True, saveImage=True):  
	image = mpimg.imread('test_images/'+file_name+'.jpg')
	print('test_images/'+file_name+'.jpg')
	# Apply calibration parameters to undistort image
	undistort_image = undistort(image)

	# Unwarp image 
	unwarp_image, M = unwarp(image)

	if saveImage==True: 
		cv2.imwrite('output_images/unwarp_'+file_name+'.jpg', cv2.cvtColor(unwarp_image, cv2.COLOR_RGB2BGR))

	# Apply thresold 
	color = color_thresolding(unwarp_image, 0)
	if saveImage==True:
		cv2.imwrite('output_images/image_processing_output/thresold_'+file_name+'.jpg', color)

	# Compute historgram for display purpose
	historgram_image = hist(color)

	# Determine right and left lane based on histogram approach
	leftx, lefty, rightx, righty, image_area = search_around_poly(color)

	# Fit polynomials
	left_fit, right_fit, left_fitx, right_fitx, ploty = fit_poly(color.shape, leftx, lefty, rightx, righty)

	# Compute radius and position on the lane
	radius, position = measure_curvature_real(color,ploty,left_fit,right_fit)


	margin = 100
	# Windows with original image and undistort image
	f, (ax1, ax2, ax3) = plt.subplots(1, 3,figsize=(24, 9))
	f.tight_layout()
	ax1.imshow(image)
	ax1.set_title('Original Image - '+file_name+'.jpg', fontsize=20)
	ax2.imshow(undistort_image)
	ax2.set_title('Undistorted image - '+file_name+'.jpg', fontsize=20)
	ax3.imshow(unwarp_image)
	ax3.set_title('Warped image - '+file_name+'.jpg', fontsize=20)
	if saveImage==True:
		f.savefig('output_images/test_images_output/undistort_unwarp_'+file_name+'.png')

	# Windows with warped image filtered image and histogram
	f1 = plt.figure(figsize=(24, 9)) 
	gs = gridspec.GridSpec(1, 3, width_ratios=[1,1,1]) 
	bx1 = plt.subplot(gs[0])
	bx1.imshow(unwarp_image)
	bx1.set_title('Warped image - '+file_name+'.jpg', fontsize=20)
	bx2 = plt.subplot(gs[1])
	bx2.imshow(color)
	bx2.set_title('Filtered image - '+file_name+'.jpg', fontsize=20)
	bx3 = plt.subplot(gs[2])
	bx3.plot(historgram_image)
	bx3.set_title('Histogram - '+file_name+'.jpg', fontsize=20)

	if saveImage==True:
		f1.savefig('output_images/test_images_output/unwarp_filter_histogram'+file_name+'.png')


	# Combine the result with the original image
	#result = cv2.addWeighted(undistort_image, 1, newwarp, 0.5, 0)
	f2 = plt.figure(figsize=(24, 9)) 
	gs = gridspec.GridSpec(1, 3, width_ratios=[1,1,1]) 
	cx1 = plt.subplot(gs[0])
	cx1.imshow(unwarp_image)
	cx1.set_title('Original image - '+file_name+'.jpg', fontsize=20)

	# Windows with polynom detection and area
	warp_zero = np.zeros_like(color).astype(np.uint8)
	color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

	# Recast the x and y points into usable format for cv2.fillPoly()
	pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
	pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
	pts = np.hstack((pts_left, pts_right))

	# Draw the lane onto the warped blank image
	color_warped = np.dstack((warp_zero, warp_zero, warp_zero))
	cv2.polylines(color_warp, np.int_([pts_left]), isClosed=False, color=(255,0,0), thickness = 40)
	cv2.polylines(color_warp, np.int_([pts_right]), isClosed=False, color=(0,0,255), thickness = 40)
	cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

	# Warp the blank back to original image space using inverse perspective matrix (Minv)
	newwarp = cv2.warpPerspective(color_warp, np.linalg.inv(M), (image.shape[1], image.shape[0])) 
	result = cv2.addWeighted(undistort_image, 1, newwarp, 0.3, 0)

	cx2 = plt.subplot(gs[1])
	cx2.imshow(image_area)
	cx2.set_title('Final image - '+file_name+'.jpg', fontsize=20)
	cx2.plot(left_fitx, ploty, color='green')
	cx2.plot(right_fitx, ploty, color='green')
	cx3 = plt.subplot(gs[2])
	# Add some text to screen curvature and lateral offset
	font                   = cv2.FONT_HERSHEY_SIMPLEX
	bottomLeftCornerOfText = (150,100)
	bottomLeftCornerOfText2 = (150,150)
	fontScale              = 1
	fontColor              = (255,255,255)
	lineType               = 2
	cv2.putText(result,'Lane curvature radius : '+str(radius), bottomLeftCornerOfText, font, fontScale, fontColor, lineType)
	cv2.putText(result,'Lateral position in lane : '+str(position), bottomLeftCornerOfText2, font, fontScale, fontColor, lineType)

	cx3.imshow(result)

	if saveImage==True:
		f2.savefig('output_images/test_images_output/original_final'+file_name+'.png')

	plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

	if show==True: 
		plt.show()

def video_pipeline(image): 
	#image = mpimg.imread('test_images/'+file_name+'.jpg')
	#print('test_images/'+file_name+'.jpg')

	# Apply calibration parameters to undistort image
	undistort_image = undistort(image)

	# Unwarp image 
	unwarp_image, M = unwarp(image)

	# Apply thresold 
	color = color_thresolding(unwarp_image, 0)

	# Compute historgram for display purpose
	historgram_image = hist(color)

	# Determine right and left lane based on histogram approach
	leftx, lefty, rightx, righty, image_area = search_around_poly(color)

	# Fit polynomials
	left_fit, right_fit, left_fitx, right_fitx, ploty = fit_poly(color.shape, leftx, lefty, rightx, righty)

	# Compute radius and position on the lane
	radius, position = measure_curvature_real(color,ploty,left_fit,right_fit)

	margin = 100

	# Windows with polynom detection and area
	warp_zero = np.zeros_like(color).astype(np.uint8)
	color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

	# Recast the x and y points into usable format for cv2.fillPoly()
	pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
	pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
	pts = np.hstack((pts_left, pts_right))

	# Draw the lane onto the warped blank image
	color_warped = np.dstack((warp_zero, warp_zero, warp_zero))
	cv2.polylines(color_warp, np.int_([pts_left]), isClosed=False, color=(255,0,0), thickness = 40)
	cv2.polylines(color_warp, np.int_([pts_right]), isClosed=False, color=(0,0,255), thickness = 40)
	cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

	# Warp the blank back to original image space using inverse perspective matrix (Minv)
	newwarp = cv2.warpPerspective(color_warp, np.linalg.inv(M), (image.shape[1], image.shape[0])) 
	result = cv2.addWeighted(undistort_image, 1, newwarp, 0.3, 0)

	# Add some text to screen curvature and lateral offset
	font                   = cv2.FONT_HERSHEY_SIMPLEX
	bottomLeftCornerOfText = (150,100)
	bottomLeftCornerOfText2 = (150,150)
	fontScale              = 1
	fontColor              = (255,255,255)
	lineType               = 2
	cv2.putText(result,'Lane curvature radius : '+str(radius), bottomLeftCornerOfText, font, fontScale, fontColor, lineType)
	cv2.putText(result,'Lateral position in lane : '+str(position), bottomLeftCornerOfText2, font, fontScale, fontColor, lineType)

	return result 


if __name__ == '__main__':
	image_pipeline('test6', show=True, saveImage=True)	

	"""white_output = 'test_videos/output.mp4'
	clip1 = VideoFileClip("test_videos/project_video.mp4")
	white_clip = clip1.fl_image(video_pipeline) #NOTE: this function expects color images!!
	white_clip.write_videofile(white_output, audio=False)"""
















	