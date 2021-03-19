from cv2 import cv2 #Importing OpenCV library
from random import randrange #Randrange for generate random colors for squares

# Load pre-trained data on face frontals form openCV 
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#Open Image
webcam =  cv2.VideoCapture(0)

while True:
    successful_frame_read , frame = webcam.read()
    grayscaled_img = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    Fc_Cor =  trained_face_data.detectMultiScale(grayscaled_img,1.4,2 )#This checks whole image for faces

    for (x,y,w,h) in Fc_Cor:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
    cv2.imshow('Face Detector',frame)
    key = cv2.waitKey(1)

    if key == 81 or key == 113:
        break

webcam.release()


print("Code Completed!")