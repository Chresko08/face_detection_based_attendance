#Tested on numpy==1.19.1 , opencv-python==4.4.0.46 , python==3.7.9

#importing openCV and operating system for database access
import cv2
import os

univ_rollno=input("Enter your university roll number:")
path="Dataset/"+univ_rollno
if not os.path.isdir(path):os.makedirs(path)

#getting the control of primary camera (0)
cam=cv2.VideoCapture(0)

#size of the window that'll open for displaying the video captured
#print("INITIALLY")
#print(cam.get(3))
#print(cam.get(4))

#changing the size of window
#print("AFTER SETTING")
cam.set(3,1280)
cam.set(4,720)
#print(cam.get(3))
#print(cam.get(4))

#Creating a cascadeClassifier object with the xml file which contains features of the face
face_features=cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")

count=0

#looping through every image frame
while True:

    #read the frame from the camera
    _,frame=cam.read()
    
    #flip the frame by y-axis
    frame = cv2.flip(frame, 1)
    
    #Converting coloured image to gray image
    gray_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    #detectMultiScale -> method to search for the face rectangle coordinates
    #scaleFactor -> decreses the shape value by 4% , smaller the value , the greater is accuracy
    #minNeighbors -> Parameter specifying how many neighbors each candidate rectangle should have to retain it.
    #In other words, this parameter will affect the quality of the detected faces.
    #Higher value results in less detections but with higher quality.
    faces=face_features.detectMultiScale(gray_frame,scaleFactor=1.5,minNeighbors=6)
    
    #print(type(faces))
    #print(faces)
    
    #adding a rectangle to the frame where face is found
    #last two parameters are colour and width of rectangle
    for x,y,w,h in faces:
        frame=cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
        count+=1
        cv2.imwrite(path+'/'+str(count) +".jpg",gray_frame[y:y+h,x:x+w])
        
    #show the frame in the window
    cv2.imshow('face recognition',frame)
    
    #it will wait for the input 'q' for 1ms , 0 means it will wait forever
    if cv2.waitKey(1)==ord('q'):
        break
        
#if 'q' input is given then close the opened window and release the camera captured
cv2.destroyWindow('face recognition')
cam.release()