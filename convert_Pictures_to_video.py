import cv2  # pip install opencv-python 
import os 
from os.path import isfile, join 

def convert_pictures_to_video(pathIn, pathOut, fps, time):
    ''' this function converts images to video'''
    frame_array=[]
    files=[f for f in os.listdir(pathIn) if isfile(join(pathIn,f))]
    for i in range (len(files)):
        filename=pathIn+files[i]
        '''reading images'''
        img=cv2.imread(filename)
        img=cv2.resize(img,(1000,800))
        height, width, layers = img.shape
        size=(width,height)
        frame_array.append(img)

    out=cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'mp4v'), fps,size)
    for i in range(len(frame_array)):
        out.write(frame_array[i])
    out.release()



#test
# directory= '/content/try'
# pathIn=directory+'/'
# pathOut=pathIn+'aavideo_EX9.avi'
# fps=1
# time=1 # the duration of each picture in the video
# convert_pictures_to_video(pathIn, pathOut, fps, time) 

'''
def pro(image):
      undist_img = undistort(image)
      return undist_img

white_output = '/content/final_result.mp4'
clip1 = VideoFileClip("/content/video/challenge_video.mp4")
white_clip = clip1.fl_image(pro) #NOTE: this function expects color images!!
%time white_clip.write_videofile(white_output, audio=False)      
'''
