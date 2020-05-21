
#importing openCV and getting the control of primary camera (0)
import cv2
cam=cv2.VideoCapture(0)

#size of the window that'll open for display the video capture
print("INITIALLY")
print(cam.get(3))
print(cam.get(4))


#changing the size of window
print("AFTER SETTING")
cam.set(3,1280)
cam.set(4,960)
print(cam.get(3))
print(cam.get(4))

while True:
    #read the frame from the camera
    bo,frame=cam.read()
    #show the frame in the window
    cv2.imshow('face recognition',frame)
    #it will wait for the input 'q' for 1ms , 0 means it will wait forever
    if cv2.waitKey(1)==ord('q'):
        break
#if 'q' input is given then close the opened window and release the camera captured
cv2.destroyWindow('face recognition')
cam.release()

