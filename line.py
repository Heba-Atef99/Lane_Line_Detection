import numpy as np
import cv2
import matplotlib.image as mpimg
from PIL import Image


# Define a class to receive the characteristics of each line detection
class Line:
    def __init__(self):
        # Set the width of the windows +/- margin
        self.window_margin = 56
        # x values of the fitted line over the last n iterations
        self.prevx = []
        # polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        #radius of curvature of the line in some units
        self.radius_of_curvature = None
        # starting x_value
        self.startx = None
        # ending x_value
        self.endx = None
        # x values for detected line pixels
        self.allx = None
        # y values for detected line pixels
        self.ally = None
        # road information
        self.road_info = None
        self.curvature = None
        self.deviation = None


def get_perspective_transform(img, src, dst, size):
    """ 
    #---------------------
    # This function takes in an image with source and destination image points,
    # generates the transform matrix and inverst transformation matrix, 
    # warps the image based on that matrix and returns the warped image with new perspective, 
    # along with both the regular and inverse transform matrices.
    #
    """

    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warp_img = cv2.warpPerspective(img, M, size, flags=cv2.INTER_LINEAR)

    return warp_img, M, Minv

def smoothing(lines, prev_n_lines=3):
    # collect lines & print average line
    """
    #---------------------
    # This function takes in lines, averages last n lines
    # and returns an average line 
    # 
    """
    lines = np.squeeze(lines)       # remove single dimensional entries from the shape of an array
    avg_line = np.zeros((720))

    for i, line in enumerate(reversed(lines)):
        if i == prev_n_lines:
            break
        avg_line += line
    avg_line = avg_line / prev_n_lines

    return avg_line

def get_lane_lines_img(binary_img, left_line, right_line):
    """
    #---------------------
    # After applying calibration, thresholding, and a perspective transform to a road image, 
    # I have a binary image where the lane lines stand out clearly. 
    # However, I still need to decide explicitly which pixels are part of the lines 
    # and which belong to the left line and which belong to the right line.
    # 
    # This get_lane_lines_img is done using histogram and sliding window
    """

    # I first take a histogram along all the columns in the lower half of the image
    histogram = np.sum(binary_img[int(binary_img.shape[0] / 2):, :], axis=0)

    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_img, binary_img, binary_img)) * 255

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] / 2)
    leftX_base = np.argmax(histogram[:midpoint])
    rightX_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    num_windows = 9
    
    # Set height of windows
    window_height = np.int(binary_img.shape[0] / num_windows)

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Current positions to be updated for each window
    current_leftX = leftX_base
    current_rightX = rightX_base

    # Set minimum number of pixels found to recenter window
    min_num_pixel = 50

    # Create empty lists to receive left and right lane pixel indices
    win_left_lane = []
    win_right_lane = []

    window_margin = left_line.window_margin

    # Step through the windows one by one
    for window in range(num_windows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_img.shape[0] - (window + 1) * window_height
        win_y_high = binary_img.shape[0] - window * window_height
        win_leftx_min = current_leftX - window_margin
        win_leftx_max = current_leftX + window_margin
        win_rightx_min = current_rightX - window_margin
        win_rightx_max = current_rightX + window_margin

        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_leftx_min, win_y_low), (win_leftx_max, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_rightx_min, win_y_low), (win_rightx_max, win_y_high), (0, 255, 0), 2)

        # Identify the nonzero pixels in x and y within the window
        left_window_inds = ((nonzeroy >= win_y_low) & (nonzeroy <= win_y_high) & (nonzerox >= win_leftx_min) & (
            nonzerox <= win_leftx_max)).nonzero()[0]
        right_window_inds = ((nonzeroy >= win_y_low) & (nonzeroy <= win_y_high) & (nonzerox >= win_rightx_min) & (
            nonzerox <= win_rightx_max)).nonzero()[0]
        # Append these indices to the lists
        win_left_lane.append(left_window_inds)
        win_right_lane.append(right_window_inds)

        # If you found > minpix pixels, recenter next window on their mean position
        if len(left_window_inds) > min_num_pixel:
            current_leftX = np.int(np.mean(nonzerox[left_window_inds]))
        if len(right_window_inds) > min_num_pixel:
            current_rightX = np.int(np.mean(nonzerox[right_window_inds]))

    # Concatenate the arrays of indices
    win_left_lane = np.concatenate(win_left_lane)
    win_right_lane = np.concatenate(win_right_lane)

    # Extract left and right line pixel positions
    leftx= nonzerox[win_left_lane]
    lefty =  nonzeroy[win_left_lane]
    rightx = nonzerox[win_right_lane]
    righty = nonzeroy[win_right_lane]

    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    left_line.current_fit = left_fit
    right_line.current_fit = right_fit

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_img.shape[0] - 1, binary_img.shape[0])
    # ax^2 + bx + c
    left_plotx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_plotx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    left_line.prevx.append(left_plotx)
    right_line.prevx.append(right_plotx)

    if len(left_line.prevx) > 10:
        left_avg_line = smoothing(left_line.prevx, 10)
        left_avg_fit = np.polyfit(ploty, left_avg_line, 2)
        left_fit_plotx = left_avg_fit[0] * ploty ** 2 + left_avg_fit[1] * ploty + left_avg_fit[2]
        left_line.current_fit = left_avg_fit
        left_line.allx, left_line.ally = left_fit_plotx, ploty
    else:
        left_line.current_fit = left_fit
        left_line.allx, left_line.ally = left_plotx, ploty

    if len(right_line.prevx) > 10:
        right_avg_line = smoothing(right_line.prevx, 10)
        right_avg_fit = np.polyfit(ploty, right_avg_line, 2)
        right_fit_plotx = right_avg_fit[0] * ploty ** 2 + right_avg_fit[1] * ploty + right_avg_fit[2]
        right_line.current_fit = right_avg_fit
        right_line.allx, right_line.ally = right_fit_plotx, ploty
    else:
        right_line.current_fit = right_fit
        right_line.allx, right_line.ally = right_plotx, ploty

    left_line.startx, right_line.startx = left_line.allx[len(left_line.allx)-1], right_line.allx[len(right_line.allx)-1]
    left_line.endx, right_line.endx = left_line.allx[0], right_line.allx[0]

    
    
    
    return out_img


def draw_lane(img, left_line, right_line, lane_color=(0, 150, 255), road_color=(255, 0, 255)):
    """ 
    #---------------------
    # This function draws lane lines and drivable area on the road
    # 
    """
    # Create an empty image to draw on
    window_img = np.zeros_like(img)

    window_margin = left_line.window_margin
    # Get the x coordinates of both the 2 left and right lane lines (driving area left and borders)
    left_lane_line_x, right_lane_line_x = left_line.allx, right_line.allx
    #get the y coordinates of the lane lines
    lane_line_y = left_line.ally
    
    """ 
    # Draw the 2 lane lines
    """
    
    # Recast the x and y points into usable format for cv2.fillPoly()
    # Make a window around the left lane line with width window_margin/8 on the left of the line and window_margin/8 on the right
    left_line_window1 = np.array([np.transpose(np.vstack([left_lane_line_x - window_margin/5, lane_line_y]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_lane_line_x + window_margin/5, lane_line_y])))])
    #concatenate the x and y points to make ordered pairs of pixels of the left lane line
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    # Make a window around the right lane line with width window_margin/8 on the left of the line and window_margin/8 on the right
    right_line_window1 = np.array([np.transpose(np.vstack([right_lane_line_x - window_margin/5, lane_line_y]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_lane_line_x + window_margin/5, lane_line_y])))])
    #concatenate the x and y points to make ordered pairs of pixels of the right lane line
    right_line_pts = np.hstack((right_line_window1, right_line_window2))
    
    # Draw the 2 lane lines onto the blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), lane_color)
    cv2.fillPoly(window_img, np.int_([right_line_pts]), lane_color)
    
    """ 
    # Draw the driving lane
    """
    
    # Recast the x and y points into usable format for cv2.fillPoly()
    driving_lane_left_pts = np.array([np.transpose(np.vstack([left_lane_line_x+window_margin/8, lane_line_y]))])
    driving_lane_right_pts = np.array([np.flipud(np.transpose(np.vstack([right_lane_line_x-window_margin/8, lane_line_y])))])
    #concatenate the x and y points to make ordered pairs of pixels of the driving lane
    driving_lane_pts = np.hstack((driving_lane_left_pts, driving_lane_right_pts))
    
    # Draw the driving lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([driving_lane_pts]), road_color)
    result = cv2.addWeighted(img, 1, window_img, 0.3, 0)

    return result, window_img

def line_curvature(left_lane, right_lane):
    """ 
    #---------------------
    # This function measures curvature of the left and right lane lines
    # in radians. 
    # This function is based on code provided in curvature measurement lecture.
    # 
    """

    ploty = left_lane.ally

    leftx, rightx = left_lane.allx, right_lane.allx

    leftx = leftx[::-1]     # Reverse to match top-to-bottom in y
    rightx = rightx[::-1]   # Reverse to match top-to-bottom in y

    # Choose y-value where we want radius of curvature which is the maximum y-value, corresponding to the bottom of the image 
    y_eval = np.max(ploty)

    # Calculate the width of the lane
    lane_width = abs(right_lane.startx - left_lane.startx)
    
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    # U.S. regulations that require a  minimum lane width of 12 feet or 3.7 meters, 
    xm_per_pix = 3.7*(720/1280) / lane_width  # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty * ym_per_pix, leftx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty * ym_per_pix, rightx * xm_per_pix, 2)
    
    # Calculate the new radii of curvature after correcting for scale in x and y
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit_cr[0])
    
    # radius of curvature result
    left_lane.radius_of_curvature = left_curverad
    right_lane.radius_of_curvature = right_curverad

def road_measurements(left_line, right_line):
    """
    #---------------------
    # This function calculates and returns follwing measurements:
    # - Radius of Curvature
    # - Distance from the Center
    # - Whether the lane is curving left or right
    # 
    """

    # Calculate the radii if the left and right lane lines
    line_curvature(left_line, right_line)

    # take average of radius of left curvature and right curvature 
    curvature = (left_line.radius_of_curvature + right_line.radius_of_curvature) / 2

    # calculate direction using X coordinates of left and right lanes 
    direction = ((left_line.endx - left_line.startx) + (right_line.endx - right_line.startx)) / 2
     
    if curvature > 2000 and abs(direction) < 100:
        curvature = -1
    
    # Calculate the centre of the lane
    center_lane = (right_line.startx + left_line.startx) / 2
    # Calculate the width of the lane 
    lane_width = right_line.startx - left_line.startx

    center_car = 720 / 2
    if center_lane > center_car:
        deviation = str(round(abs(center_lane - center_car), 3)) + 'm Left of centre'
    elif center_lane < center_car:
        deviation = str(round(abs(center_lane - center_car), 3)) + 'm Right of centre'
    else:
        deviation = 'by 0 (Centered)'
        
        
    left_line.curvature = curvature
    left_line.deviation = deviation
    
    right_line.curvature = curvature
    right_line.deviation = deviation
    

    return curvature, deviation

def illustrate_info_panel(img, left_line, right_line):
    """
    #---------------------
    # This function illustrates details below in a panel on top left corner.
    # - Lane is curving Left/Right
    # - Radius of Curvature:
    # - Deviating Left/Right by _% from center.
    #
    """

    curvature, deviation = road_measurements(left_line, right_line)
    cv2.putText(img, 'Measurements ', (75, 30), cv2.FONT_HERSHEY_COMPLEX, 0.8, (80, 80, 80), 2)

    #lane_info = 'Lane is ' + road_info
    if curvature == -1:
        lane_curve = 'Radius of Curvature : <Straight line>'
    else:
        lane_curve = 'Radius of Curvature = {0:0.3f}(m)'.format(curvature)
    deviate = 'Vehicle is ' + deviation  # deviating how much from center

    #cv2.putText(img, lane_info, (10, 63), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (100, 100, 100), 1)
    cv2.putText(img, lane_curve, (10, 83), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (100, 100, 100), 1)
    cv2.putText(img, deviate, (10, 103), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (100, 100, 100), 1)


    return img

def illustrate_driving_lane_with_topdownview(image, left_line, right_line):
    """
    #---------------------
    # This function illustrates top down view of the car on the road.
    #  
    """

    img = cv2.imread('examples/ferrari.png', -1)
    img = cv2.resize(img, (120, 246))

    rows, cols = image.shape[:2]
    window_img = np.zeros_like(image)

    window_margin = left_line.window_margin
    left_plotx, right_plotx = left_line.allx, right_line.allx
    ploty = left_line.ally
    lane_width = right_line.startx - left_line.startx
    lane_center = (right_line.startx + left_line.startx) / 2
    lane_offset = cols / 2 - (2*left_line.startx + lane_width) / 2
    car_offset = int(lane_center - 360)
    
    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([right_plotx + lane_offset - lane_width - window_margin / 4, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_plotx + lane_offset - lane_width+ window_margin / 4, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_plotx + lane_offset - window_margin / 4, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_plotx + lane_offset + window_margin / 4, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 150, 255))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 150, 255))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([right_plotx + lane_offset - lane_width + window_margin / 4, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_plotx + lane_offset - window_margin / 4, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([pts]), (255, 0, 255))

    #window_img[10:133,300:360] = img
    road_map = Image.new('RGBA', image.shape[:2], (0, 0, 0, 0))
    window_img = Image.fromarray(window_img)
    img = Image.fromarray(img)
    road_map.paste(window_img, (0, 0))
    road_map.paste(img, (300-car_offset, 590), mask=img)
    road_map = np.array(road_map)
    road_map = cv2.resize(road_map, (95, 95))
    road_map = cv2.cvtColor(road_map, cv2.COLOR_BGRA2BGR)

    return road_map

