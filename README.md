# Advanced Lane Finding


In this project, your goal is to write a software pipeline to identify the lane boundaries in a video. Compared to the first project, line detection is more successful. Several different types of filters were used, moreover the representation of the lines is based on the use of a polynomial of the second degrees, and more on a line equation.


Pipeline
---

In order to arrive at the final result, several steps were necessary. Several functions have been implemented. The various calls to these functions are combined in the `image_pipeline.py` file. 

## 1. Camera calibration

Image distorsion occurs when a cmera looks at 3D objects in real worl and transforms them into 2D images. In this first step, the goal was to determine the camera distorsion models in order to "undistort" images provided by this camera. To do that a set of chessboard images was provided. 
In the file `camera_calibration.py` the different steps of the calibration are developed. The following steps are done : 
1. Chessboard images loading 
2. Corners searching 
3. Calibration parameters computation 
4. Saving parameters
5. Parameters application

Here is some samples images (Corners searching) :

<img src="https://github.com/Dynaa/advanced-lanelines/blob/master/output_images/camera_cal_output/corners/corners_found2.jpg" width="280" alt="Corners detection Image" />  <img src="https://github.com/Dynaa/advanced-lanelines/blob/master/output_images/camera_cal_output/corners/corners_found3.jpg" width="280" alt="Corners detection Image" />  <img src="https://github.com/Dynaa/advanced-lanelines/blob/master/output_images/camera_cal_output/corners/corners_found4.jpg" width="280" alt="Corners detection Image" /> 

<img src="https://github.com/Dynaa/advanced-lanelines/blob/master/output_images/camera_cal_output/corners/corners_found5.jpg" width="280" alt="Corners detection Image" /> <img src="https://github.com/Dynaa/advanced-lanelines/blob/master/output_images/camera_cal_output/corners/corners_found7.jpg" width="280" alt="Corners detection Image" /> <img src="https://github.com/Dynaa/advanced-lanelines/blob/master/output_images/camera_cal_output/corners/corners_found8.jpg" width="280" alt="Corners detection Image" /> 

<img src="https://github.com/Dynaa/advanced-lanelines/blob/master/output_images/camera_cal_output/corners/corners_found9.jpg" width="280" alt="Corners detection Image" /> <img src="https://github.com/Dynaa/advanced-lanelines/blob/master/output_images/camera_cal_output/corners/corners_found10.jpg" width="280" alt="Corners detection Image" /> <img src="https://github.com/Dynaa/advanced-lanelines/blob/master/output_images/camera_cal_output/corners/corners_found11.jpg" width="280" alt="Corners detection Image" /> 

<img src="https://github.com/Dynaa/advanced-lanelines/blob/master/output_images/camera_cal_output/corners/corners_found12.jpg" width="280" alt="Corners detection Image" /> <img src="https://github.com/Dynaa/advanced-lanelines/blob/master/output_images/camera_cal_output/corners/corners_found13.jpg" width="280" alt="Corners detection Image" /> <img src="https://github.com/Dynaa/advanced-lanelines/blob/master/output_images/camera_cal_output/corners/corners_found14.jpg" width="280" alt="Corners detection Image" /> 

<img src="https://github.com/Dynaa/advanced-lanelines/blob/master/output_images/camera_cal_output/corners/corners_found15.jpg" width="280" alt="Corners detection Image" /> <img src="https://github.com/Dynaa/advanced-lanelines/blob/master/output_images/camera_cal_output/corners/corners_found16.jpg" width="280" alt="Corners detection Image" /> <img src="https://github.com/Dynaa/advanced-lanelines/blob/master/output_images/camera_cal_output/corners/corners_found17.jpg" width="280" alt="Corners detection Image" /> 

Given the corners found, it possible to compute undistort parameters. Those are saved in a file placed here : 
[Calibration parameters](https://github.com/Dynaa/advanced-lanelines/blob/master/camera_cal/wide_dist_pickle.p)

## 2. Application of parameters in order to undistort images

Applying the parameters allows to undistort images : 

<img src="https://github.com/Dynaa/advanced-lanelines/blob/master/output_images/camera_cal_output/undistort/undistort0.jpg" width="280" alt="Undistorted image" />  <img src="https://github.com/Dynaa/advanced-lanelines/blob/master/output_images/camera_cal_output/undistort/undistort1.jpg" width="280" alt="Undistorted image" />  <img src="https://github.com/Dynaa/advanced-lanelines/blob/master/output_images/camera_cal_output/undistort/undistort2.jpg" width="280" alt="Corners detection Image" /> 

<img src="https://github.com/Dynaa/advanced-lanelines/blob/master/output_images/camera_cal_output/undistort/undistort3.jpg" width="280" alt="Undistorted image" /> <img src="https://github.com/Dynaa/advanced-lanelines/blob/master/output_images/camera_cal_output/undistort/undistort4.jpg" width="280" alt="Undistorted image" /> <img src="https://github.com/Dynaa/advanced-lanelines/blob/master/output_images/camera_cal_output/undistort/undistort5.jpg" width="280" alt="Undistorted image" /> 

<img src="https://github.com/Dynaa/advanced-lanelines/blob/master/output_images/camera_cal_output/undistort/undistort6.jpg" width="280" alt="Undistorted image" /> <img src="https://github.com/Dynaa/advanced-lanelines/blob/master/output_images/camera_cal_output/undistort/undistort7.jpg" width="280" alt="Undistorted image" /> <img src="https://github.com/Dynaa/advanced-lanelines/blob/master/output_images/camera_cal_output/undistort/undistort8.jpg" width="280" alt="Undistorted image" /> 

<img src="https://github.com/Dynaa/advanced-lanelines/blob/master/output_images/camera_cal_output/undistort/undistort9.jpg" width="280" alt="Undistorted image" /> <img src="https://github.com/Dynaa/advanced-lanelines/blob/master/output_images/camera_cal_output/undistort/undistort10.jpg" width="280" alt="Undistorted image" /> <img src="https://github.com/Dynaa/advanced-lanelines/blob/master/output_images/camera_cal_output/undistort/undistort11.jpg" width="280" alt="Undistorted image" /> 

<img src="https://github.com/Dynaa/advanced-lanelines/blob/master/output_images/camera_cal_output/undistort/undistort12.jpg" width="280" alt="Undistorted image" /> <img src="https://github.com/Dynaa/advanced-lanelines/blob/master/output_images/camera_cal_output/undistort/undistort13.jpg" width="280" alt="Undistorted image" /> <img src="https://github.com/Dynaa/advanced-lanelines/blob/master/output_images/camera_cal_output/undistort/undistort14.jpg" width="280" alt="Undistorted image" /> 


## 3. Perspective transform

In this step, I define a region of interest on the image and then apply a transformation in order to obtain a bird-view. 
The function is based on the usage of ```cv.getPersepectiveTransform(src,dst)```, this function allow to compute a matrix M. This matrix is then used in ```cv2.warpPerspective(img, M, img.size, cv2.INTER_LINEAR)``` function. 

```python 
def unwarp(img):

	
	# Define 4 sources points : picked manually 
	left=[150,720] #left bottom
	right=[1250,720] #right bottom 
	apex_left=[590,450] # left top 
	apex_right=[700,450] # right top 

	src=np.float32([left,apex_left,apex_right,right]) # Source Points 
	dst= np.float32([[200 ,720], [200  ,0], [980 ,0], [980 ,720]]) # Destination Points 

	M = cv2.getPerspectiveTransform(src,dst)
	warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)
	return warped, M
 ```
<img src="https://github.com/Dynaa/advanced-lanelines/blob/master/output_images/test_images_output/undistort_unwarp_test1.png" width="960" alt="Undistorted image" /> 
<img src="https://github.com/Dynaa/advanced-lanelines/blob/master/output_images/test_images_output/undistort_unwarp_test2.png" width="960" alt="Undistorted image" /> 
<img src="https://github.com/Dynaa/advanced-lanelines/blob/master/output_images/test_images_output/undistort_unwarp_test3.png" width="960" alt="Undistorted image" /> 
<img src="https://github.com/Dynaa/advanced-lanelines/blob/master/output_images/test_images_output/undistort_unwarp_test4.png" width="960" alt="Undistorted image" />
<img src="https://github.com/Dynaa/advanced-lanelines/blob/master/output_images/test_images_output/undistort_unwarp_test5.png" width="960" alt="Undistorted image" />
<img src="https://github.com/Dynaa/advanced-lanelines/blob/master/output_images/test_images_output/undistort_unwarp_test6.png" width="960" alt="Undistorted image" /> 



## 4. Binary image creation

As view during the lesson, different kind of color transform and gradients were used in order to detect pixels belonging to lines. 

All theses transformations are made in the file named `image_processing.py`. The combinaison of these differents filters allow to obtain an image composed mainly by pixels related to lines. The thresolds were defined by try/results approach. 

<img src="https://github.com/Dynaa/advanced-lanelines/blob/master/test_images/test1.jpg" width="440" alt="Original image" /> <img src="https://github.com/Dynaa/advanced-lanelines/blob/master/output_images/image_processing_output/thresold_test1.jpg" width="440" alt="Filtered image" />

<img src="https://github.com/Dynaa/advanced-lanelines/blob/master/test_images/test2.jpg" width="440" alt="Original image" /> <img src="https://github.com/Dynaa/advanced-lanelines/blob/master/output_images/image_processing_output/thresold_test2.jpg" width="440" alt="Filtered image" />

<img src="https://github.com/Dynaa/advanced-lanelines/blob/master/test_images/test3.jpg" width="440" alt="Original image" /> <img src="https://github.com/Dynaa/advanced-lanelines/blob/master/output_images/image_processing_output/thresold_test3.jpg" width="440" alt="Filtered image" />

<img src="https://github.com/Dynaa/advanced-lanelines/blob/master/test_images/test4.jpg" width="440" alt="Original image" /> <img src="https://github.com/Dynaa/advanced-lanelines/blob/master/output_images/image_processing_output/thresold_test4.jpg" width="440" alt="Filtered image" />

<img src="https://github.com/Dynaa/advanced-lanelines/blob/master/test_images/test5.jpg" width="440" alt="Original image" /> <img src="https://github.com/Dynaa/advanced-lanelines/blob/master/output_images/image_processing_output/thresold_test5.jpg" width="440" alt="Filtered image" />

<img src="https://github.com/Dynaa/advanced-lanelines/blob/master/test_images/test6.jpg" width="440" alt="Original image" /> <img src="https://github.com/Dynaa/advanced-lanelines/blob/master/output_images/image_processing_output/thresold_test6.jpg" width="440" alt="Filtered image" />


## 5. Finding Lines

### 1. Histogram usage

In order to determine if pixels are related to left or right line, an Histogram was used.
<img src="https://github.com/Dynaa/advanced-lanelines/blob/master/output_images/test_images_output/unwarp_filter_histogramtest1.png" width="960" alt="Histogram from binary image" /> 
<img src="https://github.com/Dynaa/advanced-lanelines/blob/master/output_images/test_images_output/unwarp_filter_histogramtest2.png" width="960" alt="Histogram from binary image" /> 
<img src="https://github.com/Dynaa/advanced-lanelines/blob/master/output_images/test_images_output/unwarp_filter_histogramtest3.png" width="960" alt="Histogram from binary image" /> 
<img src="https://github.com/Dynaa/advanced-lanelines/blob/master/output_images/test_images_output/unwarp_filter_histogramtest4.png" width="960" alt="Histogram from binary image" /> 
<img src="https://github.com/Dynaa/advanced-lanelines/blob/master/output_images/test_images_output/unwarp_filter_histogramtest5.png" width="960" alt="Histogram from binary image" />
<img src="https://github.com/Dynaa/advanced-lanelines/blob/master/output_images/test_images_output/unwarp_filter_histogramtest6.png" width="960" alt="Histogram from binary image" /> 


1. Given an histogram, using the two peaks, we can define starting point on where the lane lines are.  
2. Given some fixed parameters, we can define a starting windows search. 
3. Looping through the number of windows we choose, for each step, determine the boundaries of our current window (based on starting point as well as margin parameter)
4. Now, determining actived pixels within the current window. 


### 2. Polynom fitting

Given the pixels belonging to left and right lines, we the fit a second order polynom using `np.polyfit`. This function allow us to find the polynom coefficients. 
test1.jpg : 
<img src="https://github.com/Dynaa/advanced-lanelines/blob/master/output_images/test_images_output/Figure_1_test1.png" width="1000" alt="Final image" />
test2.jpg : 
<img src="https://github.com/Dynaa/advanced-lanelines/blob/master/output_images/test_images_output/Figure_1_test2.png" width="1000" alt="Final image" /> 
test3.jpg : 
<img src="https://github.com/Dynaa/advanced-lanelines/blob/master/output_images/test_images_output/Figure_1_test3.png" width="1000" alt="Final image" /> 
test4.jpg : 
<img src="https://github.com/Dynaa/advanced-lanelines/blob/master/output_images/test_images_output/Figure_1_test4.png" width="1000" alt="Final image" /> 
test5.jpg : 
<img src="https://github.com/Dynaa/advanced-lanelines/blob/master/output_images/test_images_output/Figure_1_test5.png" width="1000" alt="Final image" />
test6.jpg : 
<img src="https://github.com/Dynaa/advanced-lanelines/blob/master/output_images/test_images_output/Figure_1_test6.png" width="1000" alt="Final image" /> 

### 3. Curvature and offset 

From the polynom fitting step, it's then possible to determine the curvature of the road, as well as the lateral position of the vehicle in the lane. 

#### 1. Curvature
The radius of curvature at any point xx of the function x = f(y)x=f(y) is given as follows:

<img src="https://github.com/Dynaa/advanced-lanelines/blob/master/examples/Curvature.png" width="240" alt="Curvature" /> 

In the case of the second order polynomial above, the first and second derivatives are:

<img src="https://github.com/Dynaa/advanced-lanelines/blob/master/examples/Derivatives.png" width="240" alt="Curvature" /> 

So, our equation for radius of curvature becomes:

<img src="https://github.com/Dynaa/advanced-lanelines/blob/master/examples/Curvature2.png" width="240" alt="Curvature" /> 


#### 2. Lateral position
The offset of the lane center from the center of the image (converted from pixels to meters) is the distance from the center of the lane. We detect the left line bottom and the right line bottom, given those two values, we can compute the lateral position. 

### 4. Final result on images
Once we have done all these steps, the image is unwarp and detected lines are displayed on top of the original image. The transpose of the matrix computed in the warping step is used. 

<img src="
https://github.com/Dynaa/advanced-lanelines/blob/master/output_images/test_images_output/original_finaltest1.png" width="960" alt="Final result" /> 

<img src="
https://github.com/Dynaa/advanced-lanelines/blob/master/output_images/test_images_output/original_finaltest2.png" width="960" alt="Final result" />

<img src="
https://github.com/Dynaa/advanced-lanelines/blob/master/output_images/test_images_output/original_finaltest3.png" width="960" alt="Final result" />

<img src="
https://github.com/Dynaa/advanced-lanelines/blob/master/output_images/test_images_output/original_finaltest4.png" width="960" alt="Final result" />

<img src="
https://github.com/Dynaa/advanced-lanelines/blob/master/output_images/test_images_output/original_finaltest5.png" width="960" alt="Final result" />

<img src="
https://github.com/Dynaa/advanced-lanelines/blob/master/output_images/test_images_output/original_finaltest6.png" width="240" alt="Final result" />

### 5. Application on videos

#### 1. Project video adaptation
During application of the pipeline on videos, some adaptations were needed. I used the advice to use a Line class declare in `Line.py`. 

These allow me to store line informations from previous frames. In case of non detection, it's then possible to use the lines from past images. For display purpose and readibility, an average of the last 5 curvatures as well 5 last positions ares used.

The video can be found here : [Video Challenge output](https://youtu.be/1D7dhFfJI-U)


#### 2. Challenge video adaptation 



The images for camera calibration are stored in the folder called `camera_cal`.  The images in `test_images` are for testing your pipeline on single frames.  If you want to extract more test images from the videos, you can simply use an image writing method like `cv2.imwrite()`, i.e., you can read the video in frame by frame as usual, and for frames you want to save for later you can write to an image file.  

To help the reviewer examine your work, please save examples of the output from each stage of your pipeline in the folder called `output_images`, and include a description in your writeup for the project of what each image shows.    The video called `project_video.mp4` is the video your pipeline should work well on.  

The `challenge_video.mp4` video is an extra (and optional) challenge for you if you want to test your pipeline under somewhat trickier conditions.  The `harder_challenge.mp4` video is another optional challenge and is brutal!

If you're feeling ambitious (again, totally optional though), don't stop there!  We encourage you to go out and take video of your own, calibrate your camera and show us how you would implement this project from scratch!

## How to write a README
A well written README file can enhance your project and portfolio.  Develop your abilities to create professional README files by completing [this free course](https://www.udacity.com/course/writing-readmes--ud777).

