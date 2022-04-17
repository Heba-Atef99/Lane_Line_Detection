
#frame by frame
vid_capture = cv2.VideoCapture('/content/video/project_video.mp4')
i=0
while(vid_capture.isOpened()):
    flag,frame=vid_capture.read()
    if flag==False:
        break
    #path='F:\imageProject\test_images\rame'+str(i)+'.jpg'
    path='/content/outputPhoto/frame'+str(i)+'.jpg'
    cv2.imwrite(path,frame)
    i+=1
    # Correcting for Distortion
    undist_img = undistort(frame)
    #imshow(mpimg.imread(undist_img))
    # plt.imshow(undist_img)
    # cv2.imwrite('savedimage.jpg', undist_img)   
    #io.imsave('/content/final/image'+str(i)+'.jpg',undist_img)
vid_capture.release()
cv2.destroyAllWindows()
