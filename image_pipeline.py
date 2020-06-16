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



class advanced_lane_line():
	def __init__(self, debugMode):
		# left line
		self.left_line = Line()
		# right line
		self.right_line = Line()
		# image counter
		self.image_id = 0
		# average radius values
		self.radius = []     
		# average offset
		self.offset = []
		# current radius value 
		self.current_radius = 0
		# current offset
		self.current_offset = 0
		# debug mode
		self.debug = debugMode

	def image_pipeline(self, file_name, show=True, saveImage=True):
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
		leftx, lefty, rightx, righty, left_fit, right_fit, ploty, image_area = search_around_poly(color, self.left_line, self.right_line)

		left_fitx = left_fit[0]*ploty**2+left_fit[1]*ploty+left_fit[2]
		right_fitx = right_fit[0]*ploty**2+right_fit[1]*ploty+right_fit[2]

		# Fit polynomials
		#left_fit, right_fit, left_fitx, right_fitx, ploty = fit_poly(color.shape, leftx, lefty, rightx, righty)

		# Compute radius and position on the lane
		radius, position = measure_curvature_real(color,ploty,left_fit,right_fit)

		self.radius.append(radius)
		self.offset.append(abs(position))


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

	def video_pipeline(self, image): 
		#image = mpimg.imread('test_images/'+file_name+'.jpg')
		#print('test_images/'+file_name+'.jpg')

		# Apply calibration parameters to undistort image
		undistort_image = undistort(image)

		# Unwarp image 
		unwarp_image, M = unwarp(image)

		# Apply thresold 
		color = color_thresolding(unwarp_image, 0)

		
		

		if self.debug==True : 
			cv2.imwrite('test_videos/output_images_challenge/start'+str(self.image_id)+'.jpg', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
			cv2.imwrite('test_videos/output_images_challenge/thresold_filtering'+str(self.image_id)+'.jpg', color)
			cv2.imwrite('test_videos/output_images_challenge/unwarp'+str(self.image_id)+'.jpg', cv2.cvtColor(unwarp_image, cv2.COLOR_RGB2BGR))


		# Compute historgram for display purpose
		historgram_image = hist(color)


		# Determine right and left lane based on histogram approach
		leftx, lefty, rightx, righty, left_fit, right_fit, ploty, image_area = search_around_poly(color, self.left_line, self.right_line)

		if len(left_fit) == 0 : 
			return image
		else : 

			left_fitx = left_fit[0]*ploty**2+left_fit[1]*ploty+left_fit[2]
			right_fitx = right_fit[0]*ploty**2+right_fit[1]*ploty+right_fit[2]

			if self.debug==True : 
				cv2.imwrite('test_videos/output_images_challenge/polyfit_filtering'+str(self.image_id)+'.jpg', image_area)

			# Fit polynomials
			#left_fit, right_fit, left_fitx, right_fitx, ploty = fit_poly(color.shape, leftx, lefty, rightx, righty)

			# Compute radius and position on the lane
			frame_radius, frame_position = measure_curvature_real(color,ploty,left_fit,right_fit)
			self.radius.append(frame_radius)
			self.offset.append(abs(frame_position))

			if (self.image_id%3==0):
				if len(self.radius)<5: 
					self.current_radius = np.mean(self.radius)
					self.current_offset = np.mean(self.offset)
				else: 
					self.current_radius = np.mean(self.radius[-5:-1])
					self.current_offset = np.mean(self.offset[-5:-1])

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

			cv2.putText(result,'Lane curvature radius : '+str(self.current_radius), bottomLeftCornerOfText, font, fontScale, fontColor, lineType)
			cv2.putText(result,'Lateral position in lane : '+str(self.current_offset), bottomLeftCornerOfText2, font, fontScale, fontColor, lineType)
			
			self.image_id += 1

			if self.debug==True : 
				cv2.imwrite('test_videos/output_images_challenge/result'+str(self.image_id)+'.jpg', result)

			return result 

if __name__ == '__main__':
	

	advanced_lane_line = advanced_lane_line(False)
	#advanced_lane_line.image_pipeline('start3', show=True, saveImage=True)

	white_output = 'test_videos/challenge_output_video.mp4'
	clip1 = VideoFileClip("test_videos/challenge_video.mp4")
	white_clip = clip1.fl_image(advanced_lane_line.video_pipeline) #NOTE: this function expects color images!!
	white_clip.write_videofile(white_output, audio=False)
















	