# Lane_Line_Detection




### Dependencies:

    Python 3.5.x
    NumPy
    CV2
    matplotlib
    glob
    PIL
    moviepy

### How to run the project?
Open Anaconda cmd 
Run the command ./run.sh input_video_path output_video_path mode
if mode=0 then debug mode is not deactivated and the output is a video for only the final stage
if mode=1 the debug mode is not activated and the output is a video showing all stages

### Objectives:

* Compute the camera calibration matrix and distortion coefficients using a set of chessboard images.
* Apply a distortion correction to video frames.
* Use color transforms, gradients, to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.


### (Video File References):

[project_video_in]: ./content/input_video/project_video.mp4 
[project_video_out]: ./content/project_final_result.mp4 
[challenge_video_in]: ./content/input_video/challenge_video.mp4 
[challenge_video_out]: ./content/challenge_final_result.mp4 


### Implementation Details:

[`camera_calibration.py`](camera_calibration.py) : To calculate Calibration Matrix <br />
[`line.py`](line.py) : Line class, contains functions to detect lane lines <br />
[`threshold.py`](threshold.py) : Contains functions for thresholding an image <br />
[`process.py`](process.py) : Contains the image processing pipeline and main function <br />

### step1: Camera Calibration
We applied this distortion correction to avoid camera distortion which lead to erros in calculations.
We collect objectpoint using chessboard corners which it is  same for each calibration image .

### step2:  Cropping
 We are cropping the image and resize the image to smaller dimensions to help in making the image processing pipeline faster.

### step3:  Thresholding
We used two methods of thresholding: Gradient Thresholing & HLS Thresholding. Sobel Kernel for gradient thresholding in both X and Y directions and HLS to handle cases when the road color is too bright or too light.

### step4:  Birds-Eye View
We do perspective transform to get birds-eye view .we take 4 points in a trapezoidal shape that would represent a rectangle when looking down from road above

### step5:  Sliding Window Search
We use that information and use a sliding window,placed around the line centers, to find and follow lane lines from bottom to the top of the frame.

### step6:  Radius of the curvature
 We use their x and y pixel of the lane lines

### step7:Illustrating Lane Lines on image
we illustrate the lane on the current frame, by overlaying color pixels on top of the image.




