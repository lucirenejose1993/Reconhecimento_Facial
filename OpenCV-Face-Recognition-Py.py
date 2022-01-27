#!/usr/bin/env python
# coding: utf-8

# # Face Recognition with OpenCV and Python

# In[2]:.R


#import OpenCV module
import cv2 as cv
#import os module for reading training data directories and paths
import os
#import numpy to convert python lists to numpy arrays as 
#it is needed by OpenCV face recognizers
import numpy as np


# ### Training Data

# In[3]:


#there is no label 0 in our training data so subject name for index/label 0 is empty
subjects = ["", "Arpit Dwivedi", "Udit Saxena", "Luke", "Bill Gates", "Brie Larson", "Chadwick Boseman", "Chris Evans", "Elon Musk", "Jeff Bezos", "Mark Zuckerberg", "Robert Downey",
            "Scarlett Johansson", "Steve Jobs", "Carla Diaz", "Joseph Morgan", "Sarah Andrade", "Ton Elis", "Vih Tube", "Juliette", "Julio Andrade", "Lauren German", "Nina Dobrev",
            "Pheobe Tonkin", "lei Jun", "Larry Page"]


# ### Prepare training data

# In[4]:


#function to detect face using OpenCV
def detect_face(img):
    #convert the test image to gray image as opencv face detector expects gray images
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    #load OpenCV face detector, I am using LBP which is fast
    #there is also a more accurate but slow Haar classifier
    face_cascade = cv.CascadeClassifier('opencv-files/lbpcascade_frontalface.xml')

    #let's detect multiscale (some images may be closer to camera than others) images
    #result is a list of faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5);
    
    #if no faces are detected then return original img
    if (len(faces) == 0):
        return None, None
    
    #under the assumption that there will be only one face,
    #extract the face area
    (x, y, w, h) = faces[0]
    
    #return only the face part of the image
    return gray[y:y+w, x:x+h], faces[0]


# In[5]:


#this function will read all persons' training images, detect face from each image
#and will return two lists of exactly same size, one list 
# of faces and another list of labels for each face
def prepare_training_data(data_folder_path):
    
    #------STEP-1--------
    #get the directories (one directory for each subject) in data folder
    dirs = os.listdir(data_folder_path)
    
    #list to hold all subject faces
    faces = []
    #list to hold labels for all subjects
    labels = []
    
    #let's go through each directory and read images within it
    for dir_name in dirs:
        
        #our subject directories start with letter 's' so
        #ignore any non-relevant directories if any
        if not dir_name.startswith("s"):
            continue;
            
        #------STEP-2--------
        #extract label number of subject from dir_name
        #format of dir name = slabel
        #, so removing letter 's' from dir_name will give us label
        label = int(dir_name.replace("s", ""))
        
        #build path of directory containin images for current subject subject
        #sample subject_dir_path = "training-data/s1"
        subject_dir_path = data_folder_path + "/" + dir_name
        
        #get the images names that are inside the given subject directory
        subject_images_names = os.listdir(subject_dir_path)
        
        #------STEP-3--------
        #go through each image name, read image, 
        #detect face and add face to list of faces
        for image_name in subject_images_names:
            
            #ignore system files like .DS_Store
            if image_name.startswith("."):
                continue;
            
            #build image path
            #sample image path = training-data/s1/1.pgm
            image_path = subject_dir_path + "/" + image_name

            #read image
            image = cv.imread(image_path)
            
            #display an image window to show the image 
            cv.imshow("Training on image...", image)
            cv.waitKey(100)
            
            #detect face
            face, rect = detect_face(image)
            
            #------STEP-4--------
            #for the purpose of this tutorial
            #we will ignore faces that are not detected
            if face is not None:
                #add face to list of faces
                faces.append(face)
                #add label for this face
                labels.append(label)
            
    cv.destroyAllWindows()
    cv.waitKey(1)
    cv.destroyAllWindows()
    
    return faces, labels


# In[6]:


#let's first prepare our training data
#data will be in two lists of same size
#one list will contain all the faces
#and other list will contain respective labels for each face
print("Preparing data...")
faces, labels = prepare_training_data("training-data")
print("Data prepared")

#print total faces and labels
#print("Total faces: ", len(faces))
#print("Total labels: ", len(labels))


# ### Train Face Recognizer

# In[7]:


#create our LBPH face recognizer 
recognizer = cv.face.LBPHFaceRecognizer_create()

#or use EigenFaceRecognizer by replacing above line with 
recognizer = cv.face.createEigenFaceRecognizer()

#or use FisherFaceRecognizer by replacing above line with 
recognizer = cv.face.createFisherFaceRecognizer()


# In[8]:


#train our face recognizer of our training faces
recognizer.train(faces, np.array(labels))


# ### Prediction

# In[9]:


#function to draw rectangle on image 
#according to given (x, y) coordinates and 
#given width and heigh
def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
#function to draw text on give image starting from
#passed (x, y) coordinates. 
def draw_text(img, text, x, y):
    cv.putText(img, text, (x, y), cv.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)


# In[10]:


#this function recognizes the person in image passed
#and draws a rectangle around detected face with name of the 
#subject
def predict(test_img):
    #make a copy of the image as we don't want to chang original image
    img = test_img.copy()
    #detect face from the image
    face, rect = detect_face(img)

    #predict the image using our face recognizer 
    label= recognizer.predict(face)
    #get name of respective label returned by face recognizer
    label_text = subjects[label[0]]
    
    #draw a rectangle around face detected
    draw_rectangle(img, rect)
    #draw name of predicted person
    draw_text(img, label_text, rect[0], rect[1]-5)
    
    return img


# In[ ]:


print("Predicting images...")

#load test images
test_img1 = cv.imread("test-data/1.jpg")
test_img2 = cv.imread("test-data/2.jpg")
test_img3 = cv.imread("test-data/3.jpg")
test_img4 = cv.imread("test-data/4.jpg")
test_img5 = cv.imread("test-data/5.jpg")
test_img6 = cv.imread("test-data/6.jpg")
test_img7 = cv.imread("test-data/7.jpg")
test_img8 = cv.imread("test-data/8.jpg")
test_img9 = cv.imread("test-data/9.jpg")
test_img10 = cv.imread("test-data/10.jpg")
test_img11 = cv.imread("test-data/11.jpg")
test_img12 = cv.imread("test-data/12.jpg")
test_img13 = cv.imread("test-data/13.jpg")
test_img14 = cv.imread("test-data/14.jpg")
test_img15 = cv.imread("test-data/15.jpg")
test_img16 = cv.imread("test-data/16.jpg")
test_img17 = cv.imread("test-data/17.jpg")
test_img18 = cv.imread("test-data/18.jpg")
test_img19 = cv.imread("test-data/19.jpg")
test_img20 = cv.imread("test-data/20.jpg")
test_img21 = cv.imread("test-data/21.jpg")
test_img22 = cv.imread("test-data/22.jpg")
test_img23 = cv.imread("test-data/23.jpg")
test_img24 = cv.imread("test-data/24.jpg")
test_img25 = cv.imread("test-data/25.jpg")

#perform a prediction
predicted_img1 = predict(test_img1)
predicted_img2 = predict(test_img2)
predicted_img3 = predict(test_img3)
predicted_img4 = predict(test_img4)
predicted_img5 = predict(test_img5)
predicted_img6 = predict(test_img6)
predicted_img7 = predict(test_img7)
predicted_img8 = predict(test_img8)
predicted_img9 = predict(test_img9)
predicted_img10 = predict(test_img10)
predicted_img11 = predict(test_img11)
predicted_img12 = predict(test_img12)
predicted_img13 = predict(test_img13)
predicted_img14 = predict(test_img14)
predicted_img15 = predict(test_img15)
predicted_img16 = predict(test_img16)
predicted_img17 = predict(test_img17)
predicted_img18 = predict(test_img18)
predicted_img19 = predict(test_img19)
predicted_img20 = predict(test_img20)
predicted_img21 = predict(test_img21)
predicted_img22 = predict(test_img22)
predicted_img23 = predict(test_img23)
predicted_img24 = predict(test_img24)
predicted_img25 = predict(test_img25)

print("Prediction complete")

#display both images
cv.imshow(subjects[1], predicted_img1)
cv.imshow(subjects[2], predicted_img2)
cv.imshow(subjects[3], predicted_img3)
cv.imshow(subjects[4], predicted_img3)
cv.imshow(subjects[5], predicted_img3)
cv.imshow(subjects[6], predicted_img3)
cv.imshow(subjects[7], predicted_img3)
cv.imshow(subjects[8], predicted_img3)
cv.imshow(subjects[9], predicted_img3)
cv.imshow(subjects[10], predicted_img3)
cv.imshow(subjects[11], predicted_img3)
cv.imshow(subjects[12], predicted_img3)
cv.imshow(subjects[13], predicted_img3)
cv.imshow(subjects[14], predicted_img3)
cv.imshow(subjects[15], predicted_img3)
cv.imshow(subjects[16], predicted_img3)
cv.imshow(subjects[17], predicted_img3)
cv.imshow(subjects[18], predicted_img3)
cv.imshow(subjects[19], predicted_img3)
cv.imshow(subjects[20], predicted_img3)
cv.imshow(subjects[21], predicted_img3)
cv.imshow(subjects[22], predicted_img3)
cv.imshow(subjects[23], predicted_img3)
cv.imshow(subjects[24], predicted_img3)
cv.imshow(subjects[25], predicted_img3)
cv.waitKey(0) 
cv.destroyAllWindows()


# In[ ]:





# In[ ]:




