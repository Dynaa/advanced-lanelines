import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg



def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if(orient=='x'):
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    else: 
        if(orient=='y'): 
            sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
        
    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))

    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    return sxbinary

def mag_thresh(image, sobel_kernel=3, mag_thresh=(0, 255)):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    gradmag = np.sqrt(sobelx**2+sobely**2)
    scale_factor = np.max(gradmag)/255 
    gradmag = (gradmag/scale_factor).astype(np.uint8) 
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    return binary_output

def dir_threshold(image, sobel_kernel=3, thresh=(0, np.pi/2)):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    absgradx = np.absolute(sobelx)
    absgrady = np.absolute(sobely)
    directiongrad = np.arctan2(absgrady,absgradx)
    binary_output =  np.zeros_like(directiongrad)
    binary_output[(directiongrad >= thresh[0]) & (directiongrad <= thresh[1])] = 1

    return binary_output

# Choose a Sobel kernel size
ksize = 3 # Choose a larger odd number to smooth gradient measurements


# Edit this function to create your own pipeline.
def color_thresolding(img, show, s_thresh=(180, 255), h_thresh=(15,100), b_thresh=(155,200), l_thresh=(225,255), sx_thresh=(70, 100)):
	img = np.copy(img)
	
	# Separate R channels 
	bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
	B = bgr[:,:,0]
	G = bgr[:,:,1]
	R = bgr[:,:,2]

	# Convert to HLS color space and separate channels 
	hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
	h_channel = hls[:,:,0]
	#l_channel = hls[:,:,1]
	s_channel = hls[:,:,2]

	# Convert to HLS color space and separate channels 
	luv = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
	l_channel = luv[:,:,0]
	u_channel = luv[:,:,1]
	v_channel = luv[:,:,2]

	# Yellow range 
	# yellow color mask
	lower = np.uint8([10, 0,   100])
	upper = np.uint8([40, 255, 255])
	yellow_mask = cv2.inRange(hls, lower, upper)

	# Sobel x
	sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
	abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
	scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
	# Threshold x gradient
	sxbinary = np.zeros_like(scaled_sobel)
	sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    
    # Threshold s channel
	s_binary = np.zeros_like(s_channel)
	s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    # Threshold h channel
	h_binary = np.zeros_like(h_channel)
	h_binary[(h_channel >= h_thresh[0]) & (h_channel <= h_thresh[1])] = 1

	# Thresold R channel 
	b_binary = np.zeros_like(B)
	b_binary[(R > b_thresh[0]) & (R <= b_thresh[1])] = 1

	# Thresold l channel 
	l_binary = np.zeros_like(l_channel)
	l_binary[(l_channel > l_thresh[0]) & (R <= l_thresh[1])] = 1


	# Stack each channel
	#color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary, h_binary)) * 255
	combined_binary = np.zeros_like(yellow_mask)
	combined_binary[(l_binary == 1) | (yellow_mask == 1)] = 1
	if(show==1) : 
		f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(24, 9))
		f.tight_layout()
		ax1.imshow(image)
		ax1.set_title('Original Image', fontsize=15)
		ax2.imshow(l_binary)
		ax2.set_title('l_binary', fontsize=15)
		ax3.imshow(yellow_mask)
		ax3.set_title('yellow_mask', fontsize=15)
		ax4.imshow(combined_binary, cmap='gray')
		ax4.set_title('combined_binary', fontsize=15)
		ax5.imshow(yellow_mask, cmap='gray')
		ax5.set_title('yellow_mask', fontsize=15)
		plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
		plt.show()

	return yellow_mask





if __name__ == '__main__':
	# Read in an image and grayscale it
	image = mpimg.imread('output_images/unwarp_test3.jpg')

	# Apply each of the thresholding functions
	gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=(100, 100))
	grady = abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh=(100, 100))
	mag_binary = mag_thresh(image, sobel_kernel=ksize, mag_thresh=(75, 100))
	dir_binary = dir_threshold(image, sobel_kernel=ksize, thresh=(0, np.pi/2))

	combined = np.zeros_like(dir_binary)
	combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1

	color = color_thresolding(image, 0)
	#cv2.imwrite('output_images/thresold_test3.jpg', color)


	# Run the function
	#grad_binary = abs_sobel_thresh(image, orient='x', thresh_min=20, thresh_max=100)
	# Plot the result
	