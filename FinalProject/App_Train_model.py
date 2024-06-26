import tkinter as tk
from tkinter import *
import cv2
import os
import numpy as np
from PIL import Image,ImageTk
import cv2
import numpy as np
from datetime import datetime
import csv

#####Window is our Main frame of system
window = tk.Tk()
window.title("Face Recognition Attendance System")
window.geometry('1280x720')
bg = ImageTk.PhotoImage(file = "Image.png")
label = Label(window,image = bg)
label.place(x = 0, y = 0)
window.configure()

"""
This function helps to take 200 images of an individual subject.
"""
def take_img():
    path = "dataset"
    cam = cv2.VideoCapture(0)
    detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    if not os.path.isdir(path):os.makedirs(path)
    ID = txt.get()
    Name = txt2.get()
    sampleNum = 0
    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            # incrementing sample number
            sampleNum = sampleNum + 1
            # saving the captured face in the dataset folder
            cv2.imwrite("dataset/ " + Name + "." + ID + '.' + str(sampleNum) + ".jpg",
                                gray[y:y + h, x:x + w])
            cv2.imshow('Frame', img)
                # wait for 100 miliseconds
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
                # break if the sample number is morethan 100
        elif sampleNum > 200:
            break
    cam.release()
    cv2.destroyAllWindows()
        
    res = "Images Saved  : " + ID + " Name : " + Name
    Notification.configure(text=res, bg="#d3d3d3", fg = "black",width=50, font=('times', 18, 'bold'))
    Notification.place(x=250, y=450)
      


"""
This function helps to train the model for an image
which is being taken by the take_img() function.
"""
def training():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    global detector
    detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    try:
        global faces,Id
        faces, Id = getImagesAndLabels("dataset")
    except Exception as e:
        l='please make "dataset" folder & put Images'
        Notification.configure(text=l, bg="SpringGreen3", width=50, font=('times', 18, 'bold'))
        Notification.place(x=350, y=400)

    recognizer.train(faces, np.array(Id)) 
    if not os.path.isdir("model"):os.makedirs("model")
    recognizer.save("model/trained_model2.yml")

    res = "Model Trained"  # +",".join(str(f) for f in Id)
    Notification.configure(text=res, bg="#d3d3d3", width=50, font=('times', 18, 'bold'))
    Notification.place(x=250, y=450)


def markAttendance(id,name):
    with open('Attendance.csv','r+') as f:
        myDataList = f.readlines()
        idList = []
        for line in myDataList:
            entry = line.split(',')
            idList.append(entry[2])
            sno=entry[0]
        if str(id) not in idList:
            now = datetime.now()
            current_date = now.strftime('%b-%d-%Y')
            current_time = now.strftime("%H:%M:%S")
            if sno=="S.No." : serial=1
            else : serial=int(sno)+1
            f.writelines(f'\n{serial},{name},{id},{current_date},{current_time}')

def recognise():
    """
    This function helps to take attendance and save it into an excel file 
    in the following foramt:
    Name Date
    """
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('model/trained_model2.yml')
    cascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascadePath)
    font = cv2.FONT_HERSHEY_SIMPLEX
    id = 0

    # add the list of names of your dataset here
    names = {0:'None',1734210087:'Shubham Srivastava'}

    cam = cv2.VideoCapture(0)

    while True:
        ret, img = cam.read()
        # img = cv2.flip(img, -1) # Flip vertically
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(gray,scaleFactor=1.2,minNeighbors=5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            id, confidence = recognizer.predict(gray[y:y + h, x:x + w])

            # If confidence is less them 100 ==> "0" : perfect match
            if (confidence < 100):
                stu_name = names[id]
                markAttendance(id,stu_name)
                confidence = "  {0}%".format(round(100 - confidence))
            else:
                id = "unknown"
                confidence = "  {0}%".format(round(100 - confidence))

            cv2.putText(img,stu_name,(x + 5, y - 5),font,1,(255, 255, 255),2)

        cv2.imshow('camera', img)
        k = cv2.waitKey(10) & 0xff  # Press 'ESC' for exiting video
        if k == 27:
            break
    # Do a bit of cleanup
    print("\n [INFO] Exiting Program and cleanup stuff")
    cam.release()
    cv2.destroyAllWindows()
	
def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    # create empth face list
    faceSamples = []
    # create empty ID list
    Ids = []
    # now looping through all the image paths and loading the Ids and the images
    for imagePath in imagePaths:
        # loading the image and converting it to gray scale
        pilImage = Image.open(imagePath).convert('L')
        # Now we are converting the PIL image into numpy array
        imageNp = np.array(pilImage, 'uint8')
        # getting the Id from the image

        Id = int(os.path.split(imagePath)[-1].split(".")[1])
        # extract the face from the training image sample
        faces = detector.detectMultiScale(imageNp)
        # If a face is there then append that in the list as well as Id of it
        for (x, y, w, h) in faces:
            faceSamples.append(imageNp[y:y + h, x:x + w])
            Ids.append(Id)
    return faceSamples, Ids


window.grid_rowconfigure(0, weight=1)
window.grid_columnconfigure(0, weight=1)


def on_closing():
    from tkinter import messagebox
    if messagebox.askokcancel("Quit", "Do you want to quit?"):
        window.destroy()
window.protocol("WM_DELETE_WINDOW", on_closing)


message = tk.Label(window, text="UGI, Face Recognition System", bg="#c9dfe3", fg="black",border = 0, width=50,height=2, font=('times', 30, 'italic bold '))
message.place(x=80, y=20)

Notification = Label(window, text="All things good", bg="Green", fg="white", width=15, height=3)

lbl = Label(window, text="Enter Roll No", width=20, height=2, fg="black",bg='#99badd', font=('times', 20, 'italic bold '))
lbl.place(x=200, y=250)

def testVal(inStr,acttyp):
    if acttyp == '1': #insert
        if not inStr.isdigit():
            return False
    return True

txt = tk.Entry(window, validate="key",  fg="black",font=("Calibri 20"),bg = "#aaf0d1")
txt['validatecommand'] = (txt.register(testVal),'%P','%d')	
txt.place(x=550, y=260 , width = 300, height = 40)

lbl2 = tk.Label(window, text="Enter Name", width=20, fg="black",bg='#99badd',  height=2, font=('times', 20, 'italic bold '))
lbl2.place(x=200, y=370)

txt2 = tk.Entry(window,fg="black",font=("Calibri 20"),bg ="#aaf0d1" )
txt2.place(x=550, y=380,width = 300, height = 40)

train_btn = PhotoImage(file = "images/button_train-images.png")
take_btn = PhotoImage(file = "images/button_take-image.png")
quit_btn = PhotoImage(file = "images/button_quit.png")
mark_attendance = PhotoImage(file = "images/mark_attendance.png")

takeImg = tk.Button(window , image = take_btn,command = take_img ,border=0,bg = "#F6F6F6")
takeImg.place(x=50, y=550)

trainImg = tk.Button(window, image = train_btn,command = training, border=0,bg="#F6F6F6")
trainImg.place(x=330, y=550)

recogniser = tk.Button(window,image = mark_attendance ,command=recognise, border=0,bg="#F6F6F6")
recogniser.place(x=650, y=550)

quit_Btn = tk.Button(window, image = quit_btn,command=on_closing,border=0,bg="#E8E9E4")
quit_Btn.place(x=1050, y=550)

window.mainloop()
