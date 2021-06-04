#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2, math, os
from PIL import Image
import numpy as np
import tensorflow as tf


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
                # frame = np.rot90(frame)
                #cv2.imshow('video', frame)
                numFrame += 1
                #print(numFrame)
                if (numFrame%gapFrame==0):
                    newPath = savePath + 'Frame{:04d}.png'.format(numFrame)
                    cv2.imencode('.png', frame)[1].tofile(newPath)
        else:
            break


# In[ ]:


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


# In[ ]:


# Function: Get all images from specified dir, return [None,?,?,3]
def load_images(toreadPath,split=4):
    imgList = []
    file_list = os.listdir(toreadPath)
    file_list.sort(key=lambda x:int(x[5:-split]))#倒着数第四位'.'为分界线，按照‘.’左边的数字从小到大排序
    # print(file_list)
    for imgName in file_list:
        imgPath = os.path.join(toreadPath,imgName)
        img = Image.open(imgPath)
        img_arr = np.array(img)
        imgList.append(img_arr)
    return np.array(imgList)


# In[ ]:


# BiCubic Interpolation
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
    images_list = os.listdir(imgpath)
    images_list.sort(key=lambda x:int(x[5:-4]))
    for i in images_list:
        filename = os.path.join(imgpath,i)
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


# In[ ]:


class DUF_Conv(tf.keras.models.Model):
    def __init__(self,uf):
        # 调用父类__init__()方法
        super(DUF_Conv, self).__init__()
        self.conv3d_1 = tf.keras.layers.Conv3D(64, 3, strides=1, padding='valid', activation='relu', name="conv1")
        self.rconv1a = tf.keras.layers.Conv3D(64, 1, strides=(1,1,1), padding='valid', activation='relu', name="rconv1a")
        self.rconv1b = tf.keras.layers.Conv3D(32, 3, strides=1, padding='valid', activation='relu', name="rconv1b")
        self.rconv2a = tf.keras.layers.Conv3D(96, 1, strides=(1,1,1), padding='valid', activation='relu', name="rconv2a")
        self.rconv2b = tf.keras.layers.Conv3D(32, 3, strides=1, padding='valid', activation='relu', name="rconv2b")
        self.rconv3a = tf.keras.layers.Conv3D(128, 1, strides=(1,1,1), padding='valid', activation='relu', name="rconv3a")
        self.rconv3b = tf.keras.layers.Conv3D(32, 3, strides=1, padding='valid', activation='relu', name="rconv3b")
        self.rconv4a = tf.keras.layers.Conv3D(160, 1, strides=(1,1,1), padding='valid', activation='relu', name="rconv4a")
        self.rconv4b = tf.keras.layers.Conv3D(32, 3, strides=1, padding='valid', activation='relu', name="rconv4b")
        self.rconv5a = tf.keras.layers.Conv3D(192, 1, strides=(1,1,1), padding='valid', activation='relu', name="rconv5a")
        self.rconv5b = tf.keras.layers.Conv3D(32, 3, strides=1, padding='valid', activation='relu', name="rconv5b")
        self.rconv6a = tf.keras.layers.Conv3D(224, 1, strides=(1,1,1), padding='valid', activation='relu', name="rconv6a")
        self.rconv6b = tf.keras.layers.Conv3D(32, 3, strides=1, padding='valid', activation='relu', name="rconv6b")
        self.bn1a = tf.keras.layers.BatchNormalization()
        self.bn1b = tf.keras.layers.BatchNormalization()
        self.bn2a = tf.keras.layers.BatchNormalization()
        self.bn2b = tf.keras.layers.BatchNormalization()
        self.bn3a = tf.keras.layers.BatchNormalization()
        self.bn3b = tf.keras.layers.BatchNormalization()
        self.bn4a = tf.keras.layers.BatchNormalization()
        self.bn4b = tf.keras.layers.BatchNormalization()
        self.bn5a = tf.keras.layers.BatchNormalization()
        self.bn5b = tf.keras.layers.BatchNormalization()
        self.bn6a = tf.keras.layers.BatchNormalization()
        self.bn6b = tf.keras.layers.BatchNormalization()
        # ------
        #self.fbn1 = tf.keras.layers.BatchNormalization()
        self.conv3d_s = tf.keras.layers.Conv3D(256,3,strides=(1,1,1),padding="valid", activation='relu', name="conv3d_s")
        
        self.rconv1 = tf.keras.layers.Conv3D(256,1,strides=(1,1,1),padding="valid", activation='relu', name="rconv1")
        self.rconv2 = tf.keras.layers.Conv3D(3*uf*uf,1,strides=(1,1,1),padding="valid", activation='relu', name="rconv2")
        
        self.fconv1 = tf.keras.layers.Conv3D(512,1,strides=(1,1,1),padding="valid", activation='relu', name="fconv1")
        self.fconv2 = tf.keras.layers.Conv3D(1*5*5*uf*uf,1,strides=(1,1,1),padding="valid", activation='relu', name="fconv2")
        
        self.uf = uf
        self.stp = [[0,0], [1,1], [1,1], [1,1], [0,0]]
        self.sp = [[0,0], [0,0], [1,1], [1,1], [0,0]]
        #print("Model inited.")

    def call(self, inputs):
        #print("inputs: ",inputs)
        x = self.conv3d_1(tf.pad(inputs, paddings=self.sp, mode='CONSTANT', name="padding"))
        #print("x:",x)
        
        t = self.bn1a(x)
        t = self.rconv1a(t)
        t = self.bn1b(t)
        t = self.rconv1b(tf.pad(t, paddings=self.stp, mode='CONSTANT', name="padding"))
        x = tf.concat([x, t], 4)
        #print("x:",x)
        t = self.bn2a(x)
        t = self.rconv2a(t)
        t = self.bn2b(t)
        t = self.rconv2b(tf.pad(t, paddings=self.stp, mode='CONSTANT', name="padding"))
        x = tf.concat([x, t], 4)
        #print("x:",x)
        t = self.bn3a(x)
        t = self.rconv3a(t)
        t = self.bn3b(t)
        t = self.rconv3b(tf.pad(t, paddings=self.stp, mode='CONSTANT', name="padding"))
        x = tf.concat([x, t], 4)
        #print("x:",x)
        t = self.bn4a(x)
        t = self.rconv4a(t)
        t = self.bn4b(t)
        t = self.rconv4b(tf.pad(t, paddings=self.stp, mode='CONSTANT', name="padding"))
        x = tf.concat([x, t], 4)
        #print("x:",x)
        t = self.bn5a(x)
        t = self.rconv5a(t)
        t = self.bn5b(t)
        t = self.rconv5b(tf.pad(t, paddings=self.stp, mode='CONSTANT', name="padding"))
        x = tf.concat([x, t], 4)
        #print("x:",x)
        t = self.bn6a(x)
        t = self.rconv6a(t)
        t = self.bn6b(t)
        t = self.rconv6b(tf.pad(t, paddings=self.stp, mode='CONSTANT', name="padding"))
        x = tf.concat([x, t], 4)
        #print("x:",x)
        #x = self.fbn1(x)
        x = self.conv3d_s(tf.pad(t, paddings=self.stp, mode='CONSTANT', name="padding"))
        x = tf.nn.relu(x)
        #print(x)
        
        r = self.rconv1(x)
        r = self.rconv2(r)
        #print(r)
        
        f = self.fconv1(x)
        f = self.fconv2(f)
        #print(f)
        
        ds_f = tf.shape(f)
        f = tf.reshape(f, [ds_f[0], ds_f[1], ds_f[2], ds_f[3], 25, self.uf*self.uf])
        f = tf.nn.softmax(f, axis=4)
        #print(f)

        return f,r
    
def DynFilter3D(x, F, filter_size):
    '''
    3D Dynamic filtering
    input x: (b, t, h, w)
          F: (b, h, w, tower_depth, output_depth)
          filter_shape (ft, fh, fw)
    '''
    # make tower
    filter_localexpand_np = np.reshape(np.eye(np.prod(filter_size), np.prod(filter_size)), (filter_size[1], filter_size[2], filter_size[0], np.prod(filter_size)))
    filter_localexpand = tf.Variable(filter_localexpand_np, trainable=False, dtype='float32',name='filter_localexpand') 
    x = tf.transpose(x, perm=[0,2,3,1])
    x_localexpand = tf.nn.conv2d(x, filter_localexpand, [1,1,1,1], 'SAME') # b, h, w, 1*5*5
    x_localexpand = tf.expand_dims(x_localexpand, axis=3)  # b, h, w, 1, 1*5*5
    x = tf.matmul(x_localexpand, F) # b, h, w, 1, R*R
    x = tf.squeeze(x, axis=3) # b, h, w, R*R
    return x

def depth_to_space_3D(x, block_size):
    ds_x = tf.shape(x)
    x = tf.reshape(x, [ds_x[0]*ds_x[1], ds_x[2], ds_x[3], ds_x[4]])
    
    y = tf.compat.v1.depth_to_space(x, block_size)
    
    ds_y = tf.shape(y)
    x = tf.reshape(y, [ds_x[0], ds_x[1], ds_y[1], ds_y[2], ds_y[3]])
    return x

class DUF(tf.keras.models.Model):
    def __init__(self,uf,T_in):
        # 调用父类__init__()方法
        super(DUF, self).__init__()
        self.duf_conv = DUF_Conv(uf=uf)
        self.T_in = T_in
        self.uf = uf
        #print("Model inited.")

    def call(self, inputs):
        Fx, Rx = self.duf_conv(inputs)
        #print("Fx: ",Fx)
        #print("Rx: ",Rx)
        x_c = []
        for c in range(3):
            t = DynFilter3D(inputs[:,self.T_in//2:self.T_in//2+1,:,:,c], Fx[:,0,:,:,:,:], [1,5,5]) # [B,H,W,R*R]
            t = tf.compat.v1.depth_to_space(t, self.uf) # [B,H*R,W*R,1]
            x_c += [t]
        x = tf.concat(x_c, axis=3)   # [B,H*R,W*R,3]
        x = tf.expand_dims(x, axis=1)
        Rx = depth_to_space_3D(Rx, self.uf)   # [B,1,H*R,W*R,3]
        x += Rx
        x = tf.squeeze(x, axis=1)
        #print("Out: ",x)
        return x


# In[ ]:


def con_img2v(path1,path2,svpath,axis=1,videoname="demo",fps=60):
    img_array = []
    img_list1 = os.listdir(path1)
    img_list1.sort(key=lambda x:int(x[5:-4]))
    img_list2 = os.listdir(path2)
    img_list2.sort(key=lambda x:int(x[5:-4]))
    num = min(len(img_list1),len(img_list2))
    for i in range(num):
        if i % 20 == 0:
            print("processing: %.2f%%" % (100*i/num))
        file1 = os.path.join(path1,img_list1[i])
        img1 = cv2.imread(file1)
        file2 = os.path.join(path2,img_list2[i])
        img2 = cv2.imread(file2)
        if (img1 is None) or (img2 is None):
            print(file1, file2 + " is error!")
            continue
        img = np.append(img1,img2,axis=axis)
        img_array.append(img)

    out = cv2.VideoWriter(os.path.join(svpath,'%s.mp4'%videoname), cv2.VideoWriter_fourcc('X','V','I','D'), fps, (img.shape[1],img.shape[0]))
 
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
    
# con_img2v("temp_INTER_LINEAR","temp_sr","./",1,"concat",20)


# In[ ]:




