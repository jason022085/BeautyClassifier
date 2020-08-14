# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 16:11:06 2019

@author: USER
"""

import matplotlib.pyplot as plt # plt用於顯示圖片
import matplotlib.image as mpimg # mpimg用於讀取圖片
import numpy as np 

photo = mpimg.imread('1.jpg') # 讀取和代碼處於同一目錄下的lena.png 
# 此時photo就已經是一個np.array了，可以對它進行任意處理 
#photo.shape # (512, 512, 3) 

plt.imshow(photo) # 顯示圖片 
#plt.axis( 'off' ) # 不顯示坐標軸 
#%%
photos = []
for i in range(5):
    addr = np.str(i)+'.jpg'
    new = mpimg.imread(addr)
    photos.append(new)


photos = np.array(photos)
#%%
#from keras.preprocessing.image import img_to_array, array_to_img
plt.figure(num = '高冷大姐' ,figsize=(16,16))
plt.subplot( 2,2,1)      # 將窗口分為兩行兩列四個子圖，則可顯示四幅圖片 
plt.title( ' origin image ' )    # 第一幅圖片標題 
plt.imshow(photo)

plt.subplot( 2,2,2)      # 第二個子圖 
plt.title( ' R channel ' )    # 第二幅圖片標題 
plt.imshow(photo[:,:,0],plt.cm.gray)       # 繪製第二幅圖片,且為灰度圖 
#plt.axis( ' off ' )     # 不顯示坐標尺寸

plt.subplot( 2,2,3)      # 第三個子圖 
plt.title( ' G channel ' )    # 第三幅圖片標題 
plt.imshow(photo[:,:,1],plt.cm.gray)       # 繪製第三幅圖片,且為灰度圖 
#plt.axis( ' off ' )      # 不顯示坐標尺寸

plt.subplot( 2,2,4)      # 第四個子圖 
plt.title( ' B channel ' )    # 第四幅圖片標題 
plt.imshow(photo[:,:,2],plt.cm.gray)       # 繪製第四幅圖片,且為灰度圖 
#plt.axis( ' off ' )      #不顯示坐標尺寸
#%%
plt.figure(num = '高冷大姐' ,figsize=(16,16))
plt.subplot( 2,2,1)      # 將窗口分為兩行兩列四個子圖，則可顯示四幅圖片 
plt.title( ' origin image ' )    # 第一幅圖片標題 
plt.imshow(photo)

plt.subplot( 2,2,2)      # 第二個子圖 
plt.title( ' R channel ' )    # 第二幅圖片標題 
plt.imshow(photo[:,:,0])       # 繪製第二幅圖片,且為灰度圖 
#plt.axis( ' off ' )     # 不顯示坐標尺寸

plt.subplot( 2,2,3)      # 第三個子圖 
plt.title( ' G channel ' )    # 第三幅圖片標題 
plt.imshow(photo[:,:,1])       # 繪製第三幅圖片,且為灰度圖 
#plt.axis( ' off ' )      # 不顯示坐標尺寸

plt.subplot( 2,2,4)      # 第四個子圖 
plt.title( ' B channel ' )    # 第四幅圖片標題 
plt.imshow(photo[:,:,2])       # 繪製第四幅圖片,且為灰度圖 
#plt.axis( ' off ' )      #不顯示坐標尺寸
#%%
from PIL import Image
img = Image.open('D:/Anaconda3/程式碼/ML_ishot/'+np.str(1)+'.jpg')
img1 = img.resize((200, 200),Image.ANTIALIAS)
plt.figure(num = '高冷大姐' ,figsize=(8,8))
plt.imshow(img1)
img2 = np.array(img1)
#%%
from PIL import Image
img = Image.open(r'D:/Anaconda3/mycode/PTT_photo/1.jpg')
img1 = img.resize((1000, 1000),Image.ANTIALIAS)
plt.figure(num = '高冷大姐' ,figsize=(10,10))
plt.imshow(img1)
img2 = np.array(img1)
#%%
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Album = []
for i in range(1):
    img = Image.open('D:/Anaconda3/程式碼/ML_ishot/'+np.str(i)+'.jpg').convert('LA')
    #後面的convert可以將圖片轉成灰階
    img = img.resize((300, 300),Image.ANTIALIAS)
    #plt.figure(num = np.str(i) ,figsize=(10,10))
    #plt.imshow(img)
    Album.append(np.array(img))
    print("已載入",i+1,"張")
    
print("資料輸入完成")
