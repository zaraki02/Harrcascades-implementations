#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy
import matplotlib.pyplot as plt
import pandas
import os


# In[2]:


files = os.listdir(r"C:\Users\abhij\DATA\haarcascades")
files


# In[3]:


classifier = cv2.CascadeClassifier(r"C:\Users\abhij\DATA/haarcascades\haarcascade_russian_plate_number.xml")


# In[4]:


def detect_licence(img):
    gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    coord = classifier.detectMultiScale(gray_img,1.1,5)
    return coord


# In[5]:


plate = cv2.imread(r"C:\Users\abhij\DATA\car_plate.jpg")
plate.shape
plate = cv2.cvtColor(plate,cv2.COLOR_BGR2RGB)
plt.imshow(plate[:300,:600])


# In[6]:


plate = cv2.imread(r"C:\Users\abhij\DATA\car_plate.jpg")
plate1 = cv2.imread(r"C:\Users\abhij\Downloads\car_plate1.jpg")
plate2 = cv2.imread(r"C:\Users\abhij\Downloads\car_plate2.jpg")
coord = detect_licence(plate)
print(coord)

for (x,y,w,h) in coord:
    cv2.rectangle(plate,(x,y),(x+w,y+h),(0,255,0),4)    
    
    plate[y:y+h,x:x+w] = cv2.medianBlur(plate[y:y+h,x:x+w],7) # x will be no of cols and y will be no of rows for slicing
    plt.imshow(plate)
plt.show()


# In[ ]:




