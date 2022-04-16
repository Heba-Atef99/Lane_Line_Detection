import numpy as np
import cv2

def resize_img(img):
    img = cv2.resize(img, None, fx=1/2, fy=1/2, interpolation=cv2.INTER_AREA)
    rows, cols = img.shape[:2]
    return img, rows, cols

def abs_sobel_threshold(img, orientation='x', threshold=(20, 100)):
    """
    # This function applies Sobel x or y, and then 
    # takes an absolute value and applies a threshold.
    #
    """
    # Take the derivative in x or y given orient = 'x' or 'y'
    is_orient_x = int(orientation == 'x')
    abs_sobel = cv2.Sobel(img, cv2.CV_64F, is_orient_x, int (not is_orient_x))
    abs_sobel = np.absolute(abs_sobel)

    # Scale to 8-bit (0 - 255) then convert to type = np.uint8    
    scaled_sobel = (255 * abs_sobel) / np.max(abs_sobel)
    scaled_sobel = np.uint8(scaled_sobel)

    # Create a binary mask where thresholds are met  
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= threshold[0]) & (scaled_sobel <= threshold[1])] = 255

    return binary_output

def mag_threshold(img, sobel_kernel=3, threshold=(0,255)):
    """
    # This function takes in an image and optional Sobel kernel size, 
    # as well as thresholds for gradient magnitude. And computes the gradient magnitude, 
    # applies a threshold, and creates a binary output image showing where thresholds were met.
    #
    """
    # Take the gradient in x and y separately
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    # Calculate the gradient magnitude
    grad_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

    # Scale to 8-bit (0 - 255) and convert to type = np.uint8
    grad_mag = (255 * grad_mag) / np.max(grad_mag)
    grad_mag = np.uint8(grad_mag)

    # Create a binary mask where thresholds are met  
    binary_output = np.zeros_like(grad_mag)
    binary_output[(grad_mag >= threshold[0]) & (grad_mag <= threshold[1])] = 255
    return binary_output

def dir_threshold(img, sobel_kernel=3, threshold=(0.7,1.3)):
    """
    # This function takes in an image and optional Sobel kernel size, 
    # as well as thresholds for gradient magnitude. Then computes the direction of the gradient, 
    # applies a threshold, and creates a binary output image showing where thresholds were met.
    #
    """
    # Take the gradient in x and y separately
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    # Calculate the gradient magnitude
    grad_dir = np.arctan2(np.absolute(sobel_x), np.absolute(sobel_y))

    # Create a binary mask where thresholds are met  
    binary_output = np.zeros_like(grad_dir)
    binary_output[(grad_dir >= threshold[0]) & (grad_dir <= threshold[1])] = 255
    return binary_output


def get_combined_gradients(img, thresh_x, thresh_y, thresh_mag, thresh_dir):
    """
    # This function isolates lane line pixels, by focusing on pixels
    # that are likely to be part of lane lines.
    # I am using Red Channel, since it detects white pixels very well. 
    """
    rows = img.shape[0]
    # cv2.imshow("name23", img)

    # crop the image to focus on the lanes more
    # crop till rows-12 so that the car part doesn't appear
    R_channel = img[220:rows-12, :, 2]
    # cv2.imshow("croped", R_channel)

    sobel_x = abs_sobel_threshold(R_channel, 'x', thresh_x)
    sobel_y = abs_sobel_threshold(R_channel, 'y', thresh_y)
    # cv2.imshow("sob_x", sobel_x)
    # cv2.imshow("sob_y", sobel_y)
    mag_binary = mag_threshold(R_channel, 3, thresh_mag)
    # cv2.imshow("mag", mag_binary)
    dir_binary = dir_threshold(R_channel, 15, thresh_dir)
    # cv2.imshow("dir", dir_binary)

    # combine sobelx, sobely, magnitude & direction measurements
    gradient_combined = np.uint8(np.zeros_like(dir_binary))
    gradient_combined[((sobel_x > 1) & (mag_binary > 1) & (dir_binary > 1)) | ((sobel_x > 1) & (sobel_y > 1))] = 255
    # cv2.imshow("gc", gradient_combined)
    return gradient_combined

if __name__ == "__main__":
    th_sobelx, th_sobely, th_mag, th_dir = (35, 100), (30, 255), (30, 255), (0.7, 1.3)
    th_h, th_l, th_s = (10, 100), (0, 60), (85, 255)

    img_path = "../advanced-lane-detection-for-self-driving-cars-master/output_images/01_undist_img.png"
    # img_path = "test_images/test6.jpg"
    img = cv2.imread(img_path)
    # print(img.shape)
    # cv2.imshow("name1", img)
    undistorted_img, rows, cols = resize_img(img)

    combined_gradient = get_combined_gradients(undistorted_img, th_sobelx, th_sobely, th_mag, th_dir)
    # So that the window won't close immediatly
    cv2.waitKey(0)


# RESOURCES
# cv2.resize ==> https://www.tutorialkart.com/opencv/python/opencv-python-resize-image/
# cv2.Sobel ==> https://docs.opencv.org/3.4/d2/d2c/tutorial_sobel_derivatives.html
# np.arctan2 ==> https://numpy.org/doc/stable/reference/generated/numpy.arctan2.html