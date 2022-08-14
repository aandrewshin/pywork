import cv2

#Load pre-trained data from opencv xml file
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#choose image to detect face 
img = cv2.imread('rdj.png') #this is how you upload an image

#make img gray so colour is one number and not rgb which is easier for program
gscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 
#detect Faces
face_coordinates = trained_face_data.detectMultiScale(gscale_img)

#draw rectangle around face 
(x,y,w,h) = face_coordinates[0]
cv2.rectangle(img,(x, y),(x+w, y+h), (0,0,255),7 )


cv2.imshow('Face Detector' , img)
 #used to wait before terminating program. without this, image is shown for a second and continues to end the program
 #keeps image open
cv2.waitKey()

print ("Code Completed")