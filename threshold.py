from pickletools import uint8
from unittest import result
import numpy as np
import cv2

def crop(img):
    rows = img.shape[0]
    start_crop = int(11*rows/18)
    img = img[start_crop:rows-12, :, :]
    return img

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
    img = crop(img)

    # crop the image to focus on the lanes more
    # crop till rows-12 so that the car part doesn't appear
    R_channel = img[:,:,2]

    sobel_x = abs_sobel_threshold(R_channel, 'x', thresh_x)
    sobel_y = abs_sobel_threshold(R_channel, 'y', thresh_y)

    mag_binary = mag_threshold(R_channel, 3, thresh_mag)
    dir_binary = dir_threshold(R_channel, 15, thresh_dir)

    gradient_combined = np.uint8(np.zeros_like(dir_binary))
    gradient_combined[((sobel_x > 1) & (mag_binary > 1) & (dir_binary > 1)) | ((sobel_x > 1) & (sobel_y > 1))] = 255
    return gradient_combined

def channel_threshold(channel, threshold=(80, 255)):
    """
    # This function takes in a channel of an image and
    # returns thresholded binary image
    # 
    """
    binary_output = np.zeros_like(channel)
    binary_output[(channel > threshold[0]) & (channel <= threshold[1])] = 255
    return binary_output


def get_combined_hls(img, th_h, th_l, th_s):
    """
    # This function takes in an image, converts it to HLS colorspace, 
    # extracts individual channels, applies thresholding on them
    #
    """
    img = crop(img)

    # convert img to hls color space
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

    h = hls[:, :, 0]
    l = hls[:, :, 1]
    s = hls[:, :, 2]

    h_channel = channel_threshold(h, th_h)
    l_channel = channel_threshold(l, th_l)
    s_channel = channel_threshold(s, th_s)

    hls_combined = np.uint8(np.zeros_like(s_channel))
    hls_combined[((s_channel > 1) & (l_channel == 0)) | ((s_channel == 0) & (h_channel > 1) & (l_channel > 1))] = 255
    return hls_combined

def combine_grad_hls(combined_grad, combined_hls):
    """ 
    # This function combines gradient and hls images into one.
    # For binary gradient image, if pixel is bright, set that pixel value in reulting image to 255
    # For binary hls image, if pixel is bright, set that pixel value in resulting image to 255 
    # Edit: Assign different values to distinguish them
    # 
    """

    r, c = combined_grad.shape
    half_c = int(c/2)
    diff = c - 2 * half_c
    zeros_arr = np.zeros((r, half_c), np.uint8)
    ones_arr = np.ones((r, half_c + diff), np.uint8)
    mask = np.concatenate((zeros_arr, ones_arr), axis=1)
    combined_grad = combined_grad * mask

    result = np.uint8(np.zeros_like(combined_hls))
    result[(combined_grad>1)] = 255
    result[combined_hls>1] = 255
    return result

# RESOURCES
# cv2.resize ==> https://www.tutorialkart.com/opencv/python/opencv-python-resize-image/
# cv2.Sobel ==> https://docs.opencv.org/3.4/d2/d2c/tutorial_sobel_derivatives.html
# np.arctan2 ==> https://numpy.org/doc/stable/reference/generated/numpy.arctan2.html