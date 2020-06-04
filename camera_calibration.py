import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import argparse
import glob

# Number of inside corners in x and y 
nx = 9
ny = 6
img_size = (720,1280)

def findChessboardCorners(path, objpoints, imgpoints):
    images = glob.glob('camera_cal/calibration*.jpg')
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    

    # Step through the list and search for chessboard corners
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
        print(fname)
        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Draw and display the corners
            cv2.drawChessboardCorners(img, (9,6), corners, ret)
            write_name = 'output_images/camera_cal_output/corners/corners_found'+str(idx)+'.jpg'
            cv2.imwrite(write_name, img)
            #cv2.imshow('img', img)
            cv2.waitKey(500)

    cv2.destroyAllWindows()


def calibration(objpoints, imgpoints): 
    # Do camera calibration given object points and image points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)
    
    # Save the camera clibration result for later use 
    dist_pickle = {}
    dist_pickle["mtx"] = mtx
    dist_pickle["dist"] = dist
    pickle.dump( dist_pickle, open( "camera_cal/wide_dist_pickle.p", "wb" ) )

def undistort(img): 
    #images = glob.glob('camera_cal/calibration*.jpg')
    dist_pickle = pickle.load( open( "camera_cal/wide_dist_pickle.p", "rb" ) )
    mtx = dist_pickle["mtx"]
    dist = dist_pickle["dist"]

    #for idx, fname in enumerate(images):
    #img = cv2.imread(fname)
    dst = cv2.undistort(img,mtx,dist,None, mtx)
    write_name = 'output_images/test_images_output/undistort.jpg'
    cv2.imwrite(write_name, cv2.cvtColor(dst, cv2.COLOR_RGB2BGR))
    return dst

if __name__ == '__main__':
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.
    parser = argparse.ArgumentParser('Define camera calibration parameters')
    parser.add_argument('-f', default='camera_cal/calibration1.jpg', help='path to test image')
    args = parser.parse_args()
    print(args.f)
    findChessboardCorners(args.f, objpoints, imgpoints)
    calibration(objpoints,imgpoints)
    #undistort(image)
