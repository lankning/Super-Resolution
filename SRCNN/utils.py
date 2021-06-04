#!/usr/bin/env python
# coding: utf-8

# In[2]:


import cv2, math, os
from PIL import Image
import numpy as np


# In[ ]:


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
                #cv2.imshow('video', frame)
                numFrame += 1
                #print(numFrame)
                if (numFrame%gapFrame==0):
                    newPath = savePath + str(numFrame) + ".jpg"
                    cv2.imencode('.jpg', frame)[1].tofile(newPath)
        else:
            break

# getFrame("./data/liziqi.mp4", "./data/x_train_data/", 600)


# In[8]:


# def conv_2d(matrix, stride, mode="average"):
#     a,b = matrix.shape
#     conv_kernel = np.zeros((stride,stride))
#     if mode=="average":
#         conv_kernel[:][:] = 1/stride
#     elif mode == "select":
#         conv_kernel[0][0] = 1
    
#     if (a%stride)!=0:
#         temp = np.zeros((stride-a%stride,b))
#         matrix = np.concatenate((matrix,temp),axis=0)
#     a,b = matrix.shape
#     if (b%stride)!=0:
#         temp = np.zeros((a,stride-b%stride))
#         matrix = np.concatenate((matrix,temp),axis=1)
#     a,b = matrix.shape
#     newmat = np.zeros((int(a/stride),int(b/stride)))
#     for i in range(int(a/stride)):
#         for j in range(int(b/stride)):
#             newmat[i,j] = np.sum(np.multiply(matrix[stride*i:stride*i+stride,stride*j:stride*j+stride],conv_kernel))
#     return newmat


# def blurry(hrPath, savePath, blurryTimes=4, mode="average"):
#     img = Image.open(hrPath)
#     img_arr = np.array(img)
#     stride = int(math.sqrt(blurryTimes))
#     d0 = conv_2d(img_arr[:,:,0],stride,mode)
#     d1 = conv_2d(img_arr[:,:,1],stride,mode)
#     d2 = conv_2d(img_arr[:,:,2],stride,mode)
#     a,b = d0.shape
#     # print(a,b)
#     newmat = np.zeros((a,b,3))
#     newmat[:,:,0] = d0
#     newmat[:,:,1] = d1
#     newmat[:,:,2] = d2
#     img = Image.fromarray(np.uint8(newmat),"RGB")
#     filename = (hrPath.split('/')[-1]).split('.')[0]
#     newPath = savePath + filename + "_" + str(blurryTimes) + "x.jpg"
#     img.save(newPath)
# # blurry("./data/x_train_data/600.jpg","./data/x_train_data4x/",4,"select")

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

# blurryList("./data/train_data/","./data/train_data4x/",4)#,"select"


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


# In[15]:


# BiCubic Interpolation
# this code is original from 
# https://blog.csdn.net/qq_34885184/article/details/79163991
# However, Pillow.Image.BICUBIC() can be used.
# def sw(w1):
#     w = abs(w1)
#     a=-0.5
#     if(w<1 and w>0):
#         A=1-(a+3)*(w**2)+(a+2)*(w**3)
#     elif(w>1 and w<2):
#         A=a*(w**3)-5*a*(w**2)+(8*a)*w-4*a;
#     else:
#         A=0
#     return A

# def BiCubic(img_arr,k):
#     #Something wrong
#     a,b,c = img_arr.shape
#     temp = np.zeros((2,b,c))
#     img_arr = np.concatenate((img_arr,temp),axis=0)
#     img_arr = np.concatenate((temp,img_arr),axis=0)
#     temp = np.zeros((a+4,2,c))
#     img_arr = np.concatenate((img_arr,temp),axis=1)
#     img_arr = np.concatenate((temp,img_arr),axis=1)
#     bigImg = np.zeros((a*k,b*k,c))
#     for d in range(c):
#         for i in range(k*a):
#             u = (i%k)/k
#             i1 = int(i/k)+2
#             A = np.array([[sw(1+u),sw(u),sw(1-u),sw(2-u)]])
#             for j in range(k*b):
#                 v = (j%k)/k
#                 j1 = int(j/k)+2
#                 C = np.array([[sw(1+v),sw(v),sw(1-v),sw(2-v)]])
#                 B = np.array([
#                     [img_arr[i1-1,j1-1,d],img_arr[i1-1,j1,d],img_arr[i1-1,j1+1,d],img_arr[i1-1,j1+2,d]],
#                     [img_arr[i1,j1-1,d],img_arr[i1,j1,d],img_arr[i1,j1+1,d],img_arr[i1,j1+2,d]],
#                     [img_arr[i1+1,j1-1,d],img_arr[i1+1,j1,d],img_arr[i1+1,j1+1,d],img_arr[i1+1,j1+2,d]],
#                     [img_arr[i1+2,j1-1,d],img_arr[i1+2,j1,d],img_arr[i1+2,j1+1,d],img_arr[i1+2,j1+2,d]],
#                              ])
#                 temp0 = np.dot(B,A.T)
#                 temp1 = np.dot(C,temp0)
#                 bigImg[i,j,d] = temp1[0,0]
#     return bigImg

def BiCubicList(imgdata,k):
    num,a,b,c = imgdata.shape
    bigImgs = np.zeros((num,k*a,k*b,c))
    for i in range(num):
        # use the implementation above
        # bigImgs.append(BiCubic(imgdata[i,...],k))
        # For efficiency, we use Pillow.Image.BICUBIC() method
        rgb = Image.fromarray(np.uint8(imgdata[i,...]),"RGB")
        rgb = rgb.resize((b*k,a*k), Image.BICUBIC)
        bigImgs[i,...] = np.array(rgb)
    return bigImgs


# In[ ]:


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


# In[ ]:


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


# In[ ]:


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

