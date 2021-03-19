from cv2 import cv2 #Importing OpenCV library
from random import randrange #Randrange for generate random colors for squares

#Load pre-trained data on face frontals form openCV 
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#Open Image
img = cv2.imread('1.jpg') 
#Image must be converted to grayscale
grayscaled_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#Detect Faces
Fc_Cor =  trained_face_data.detectMultiScale(grayscaled_img)#This checks whole image for faces

#Loop for every face and draw rectangles
for (x,y,w,h) in Fc_Cor:
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,randrange(255),randrange(255)),10)

imS = cv2.resize(img, (960, 540)) #Resize the image

cv2.imshow('Face Detector',imS)
cv2.waitKey()





print("Code Completed!")