from statistics import mode
import sys
from moviepy.editor import VideoFileClip
import numpy as np
import cv2
from camera_calibration import calib, undistort
from threshold import get_combined_gradients, get_combined_hls, combine_grad_hls
from line import Line, get_perspective_transform, get_lane_lines_img, draw_lane, illustrate_info_panel, illustrate_driving_lane_with_topdownview

#tune the parameters
th_sobelx, th_sobely, th_mag, th_dir = (30, 100), (30, 100), (70, 100), (np.pi/5, np.pi/2)
th_h, th_l, th_s = (39, 100), (0, 60), (100, 255)

left_line = Line()
right_line = Line()
    
mtx, dist = calib()

#pipeline
def main_pipeline(frame):

    # ****** Stage1: Correcting for Distortion ****** 
    undist_img = undistort(frame, mtx, dist)
    # undist_img = undistort(frame)

    # resize frame for faster processing
    undist_img = cv2.resize(undist_img, None, fx=1 / 2, fy=1 / 2, interpolation=cv2.INTER_AREA)
    rows, cols = undist_img.shape[:2]

    # ****** Stage2: Thresholding ****** 
    combined_gradient = get_combined_gradients(undist_img, th_sobelx, th_sobely, th_mag, th_dir)
    combined_hls = get_combined_hls(undist_img, th_h, th_l, th_s)
    combined_result = combine_grad_hls(combined_gradient, combined_hls)

    c_rows, c_cols = combined_result.shape[:2]
    s_LTop2, s_RTop2 = [c_cols / 2 - 24, 5], [c_cols / 2 + 24, 5]
    s_LBot2, s_RBot2 = [110, c_rows], [c_cols - 110, c_rows]

    src = np.float32([s_LBot2, s_LTop2, s_RTop2, s_RBot2])
    dst = np.float32([(170, 720), (170, 0), (550, 0), (550, 720)])

    # ****** Stage3: Warped(Bird Eye View) ****** 
    warp_img, M, Minv = get_perspective_transform(combined_result, src, dst, (720, 720))

    # ****** Stage4: Sliding window search  ****** 
    searching_img = get_lane_lines_img(warp_img, left_line, right_line)

    # ****** Stage5: Illustrat lane****** 
    w_comb_result, w_color_result = draw_lane(searching_img, left_line, right_line)

    # Drawing the lines back down onto the road
    color_result = cv2.warpPerspective(w_color_result, Minv, (c_cols, c_rows))
    lane_color = np.zeros_like(undist_img)
    lane_color[220:rows - 12, 0:cols] = color_result

    # Combine the result with the original image
    result = cv2.addWeighted(undist_img, 1, lane_color, 0.3, 0)

    info_panel, birdeye_view_panel = np.zeros_like(result),  np.zeros_like(result)
    info_panel[5:110, 5:325] = (255, 255, 255)
    birdeye_view_panel[5:110, cols-111:cols-6] = (255, 255, 255)
    
    info_panel = cv2.addWeighted(result, 1, info_panel, 0.2, 0)
    birdeye_view_panel = cv2.addWeighted(info_panel, 1, birdeye_view_panel, 0.2, 0)
    road_map = illustrate_driving_lane_with_topdownview(w_color_result, left_line, right_line)
    birdeye_view_panel[10:105, cols-106:cols-11] = road_map
    birdeye_view_panel = illustrate_info_panel(birdeye_view_panel, left_line, right_line)
    
    return birdeye_view_panel, combined_result, searching_img

def pipeline_0(frame):
    birdeye_view_panel, _ , _ = main_pipeline(frame)
    return birdeye_view_panel

def pipeline_1(frame):
    birdeye_view_panel, combined_result, searching_img= main_pipeline(frame)
    combined_result = cv2.cvtColor(combined_result, cv2.COLOR_GRAY2BGR)
    combined_result = cv2.resize(combined_result, (searching_img.shape[1], 200))

    combined_image = np.vstack([combined_result, searching_img])
    combined_image = cv2.resize(combined_image, (300, birdeye_view_panel.shape[0]))
    combined_image = np.hstack([birdeye_view_panel, combined_image])
    return combined_image

#the main
def main():

    args=sys.argv
    input_path=args[1]
    output_path=args[2]
    modes=args[3]

    # project_video_path = 'content/input_video/project_video.mp4'
    # challenge_video_path = 'content/input_video/challenge_video.mp4'
    # output_project_vid_path = 'content/project_final_result.mp4'
    # output_challenge_vid_path = 'content/challenge_final_result.mp4'

    #generate the output video
    white_output = output_path
    clip1 = VideoFileClip(input_path)
    if modes == '0':
        white_clip = clip1.fl_image(pipeline_0)
    elif modes == '1':
        white_clip = clip1.fl_image(pipeline_1)
    else:
        print("Mode shall be 0 or 1 only")
    white_clip.write_videofile(white_output, audio=False) 

if __name__ == '__main__':
    main()
