#!/usr/bin/env python
# coding: utf-8

# In[2]:


import cv2, math, os
from PIL import Image
import numpy as np
import tensorflow as tf


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
                #cv2.imshow('video', frame)
                numFrame += 1
                #print(numFrame)
                if (numFrame%gapFrame==0):
                    newPath = savePath + 'Frame{:04d}.png'.format(numFrame)
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


# In[4]:


# Function: Get all images from specified dir, return [None,?,?,3]
def load_images(toreadPath,split=4):
    imgList = []
    file_list = os.listdir(toreadPath)
    file_list.sort(key=lambda x:int(x[:-split]))#倒着数第四位'.'为分界线，按照‘.’左边的数字从小到大排序
    # print(file_list)
    for imgName in file_list:
        imgPath = os.path.join(toreadPath,imgName)
        img = Image.open(imgPath)
        img_arr = np.array(img)
        imgList.append(img_arr)
    return np.array(imgList)


# In[5]:


def BiCubicList(imgdata,k):
    num,a,b,c = imgdata.shape
    bigImgs = np.zeros((num,k*a,k*b,c))
    for i in range(num):
        # For efficiency, we use Pillow.Image.BICUBIC() method
        rgb = Image.fromarray(np.uint8(imgdata[i,...]),"RGB")
        rgb = rgb.resize((b*k,a*k), Image.BICUBIC)
        bigImgs[i,...] = np.array(rgb)
    return bigImgs


# In[6]:


# Function: Get all images from specified dir, return [None,?,?,3]
def load_images_bybicubic(toreadPath,k=2,split=4):
    imgList = []
    file_list = os.listdir(toreadPath)
    file_list.sort(key=lambda x:int(x[:-split]))#倒着数第四位'.'为分界线，按照‘.’左边的数字从小到大排序
    # print(file_list)
    for imgName in file_list:
        imgPath = os.path.join(toreadPath,imgName)
        img = Image.open(imgPath)
        img_arr = np.array(img)
        a,b,c = img_arr.shape
        rgb = Image.fromarray(np.uint8(img_arr),"RGB")
        rgb = rgb.resize((b*k,a*k), Image.BICUBIC)
        img_arr = np.array(rgb)
        imgList.append(img_arr)
    return np.float32(np.array(imgList))


# In[7]:


def loadimgs_from_paths(x_paths,y_paths):
    xList=[]
    for x_path in x_paths:
        img = Image.open(x_path)
        img_arr = np.array(img)
        xList.append(img_arr)
    yList=[]
    for y_path in y_paths:
        img = Image.open(y_path)
        img_arr = np.array(img)
        yList.append(img_arr)
    return np.array(xList),np.array(yList)


# In[8]:


def images_to_video(imgpath,svpath,videoname="demo",fps=60):
    img_array = []
    file_list = os.listdir(imgpath)
    file_num = len(file_list)
    for i in range(file_num):
        filename = os.path.join(imgpath,str(i+1))+".jpg"
        print(filename)
        img = cv2.imread(filename)
        if img is None:
            print(filename + " is error!")
            continue
        img_array.append(img)

    out = cv2.VideoWriter(os.path.join(svpath,'%s.avi'%videoname), cv2.VideoWriter_fourcc('X','V','I','D'), fps, (img.shape[1],img.shape[0]))
 
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()


# In[9]:


class PixelShuffler(tf.keras.layers.Layer):
    def __init__(self, target_shape, **kwargs):
        super(PixelShuffler, self).__init__(**kwargs)
        self.target_shape = tuple(target_shape)
        print("init succeed.")
    
    def build(self,input_shape):
        # self.inshape = input_shape
        print("build succeed.")

    def call(self, inputs,color=True):
        bsize,inh,inw,ind = inputs.shape
        if bsize is None:
                bsize = -1
        h,w,d = self.target_shape
        if ((h%inh!=0) or (w%inw!=0) or (ind%d!=0) or ((h/inh)!=(w/inw)) or ((h/inh)*(w/inw)!=(ind/d))):
            raise Exception("Error! The shape of input and target is not corresponded.")
        r = int(h/inh)
        result = []
        if color:
            Xc = tf.split(inputs, 3, 3)
            for x in Xc:
                X = tf.reshape(x, (bsize, inh, inw, r, r))
                X = tf.transpose(X, (0, 1, 2, 4, 3))  # bsize, a, b, 1, 1
                X = tf.split(X, inh, 1)  # a, [bsize, b, r, r]
                X = tf.concat([tf.squeeze(x) for x in X],2)  # bsize, b, a*r, r
                X = tf.split(X, inw, 1)  # b, [bsize, a*r, r]
                X = tf.concat([tf.squeeze(x) for x in X],2)  # bsize, a*r, b*r
                X = tf.reshape(X, (bsize, inh*r, inw*r, 1))
                result.append(X)
            X = tf.concat(result,3)
        else:
            X = tf.reshape(inputs, (bsize, inh, inw, r, r))
            X = tf.transpose(X, (0, 1, 2, 4, 3))  # bsize, a, b, 1, 1
            X = tf.split(X, 1, inh)  # a, [bsize, b, r, r]
            X = tf.concat([tf.squeeze(x) for x in X],2)  # bsize, b, a*r, r
            X = tf.split(X, 1, inw)  # b, [bsize, a*r, r]
            X = tf.concat([tf.squeeze(x) for x in X],2)  # bsize, a*r, b*r
            X = tf.reshape(X, (bsize, inh*r, inw*r, 1))
            X = tf.concat(result,3)
        print("PixelShuffle Finished.")
        return X

