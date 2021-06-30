#!/usr/bin/env python
# coding: utf-8

# In[2]:


import cv2, math, os
from PIL import Image
import numpy as np


# In[2]:


# Function: Get frames from video with interval(gapFrame)
def getFrame(videoPath, savePath, gapFrame=1):
# this code is original from https://blog.csdn.net/u010555688/article/details/79182362
    cap = cv2.VideoCapture(videoPath)
    numFrame = 0
    while True:
        if cap.grab():
            flag, frame = cap.retrieve()
            if not flag:
                continue
            else:
                # frame = np.rot90(frame)
                # cv2.imshow('video', frame)
                numFrame += 1
                #print(numFrame)
                if (numFrame%gapFrame==0):
                    newPath = savePath + 'Frame{:06d}.png'.format(numFrame)
                    cv2.imencode('.png', frame)[1].tofile(newPath)
        else:
            break


# In[3]:


# get LR pics from HR pics by 1/blurryTimes with mode "select"
def blurryList(imgFolder, svFolder, blurryTimes=4):#, mode="select"
    img_list = os.listdir(imgFolder)
    for imgName in img_list:
        imgPath = os.path.join(imgFolder,imgName)
        # blurry(imgPath,svFolder,blurryTimes,mode)
        img = cv2.imread(imgPath, cv2.IMREAD_COLOR)
        w,h,d = img.shape
        img = cv2.resize(img, (h//blurryTimes, w//blurryTimes),interpolation=cv2.INTER_NEAREST)
        svpath = os.path.join(svFolder,imgName)
        cv2.imwrite(svpath, img)


# In[ ]:


def CutList(imgFolder, svFolder,left=1/3,right=2/3,up=1/3,down=2/3):
    img_list = os.listdir(imgFolder)
    for imgName in img_list:
        imgPath = os.path.join(imgFolder,imgName)
        # blurry(imgPath,svFolder,blurryTimes,mode)
        img = cv2.imread(imgPath, cv2.IMREAD_COLOR)
        h,w,_ = np.shape(img)
        img = img[int(h*up):int(h*down),int(w*left):int(w*right),:]
        svpath = os.path.join(svFolder,imgName)
        cv2.imwrite(svpath, img)


# In[ ]:




