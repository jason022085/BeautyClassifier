# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 15:38:08 2019

@author: Chang-Yo H.F.
"""

import matplotlib.pyplot as plt # plt用於顯示圖片
import matplotlib.image as mpimg # mpimg用於讀取圖片
import numpy as np
from PIL import Image
from PIL import ImageFile
import tensorflow as tf
import pandas as pd
#%%

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Album = []
for i in range(924):
    img = Image.open('D:/Anaconda3/程式碼/ML_ishs/'+'m ('+np.str(i+1)+').jpg').convert('L')    #後面的convert可以將圖片轉成灰階
    img = img.resize((400, 400),Image.ANTIALIAS)
    #plt.figure(num = np.str(i) ,figsize=(10,10))
    #plt.imshow(img)
    img = np.array(img)
    img = img/255
    Album.append(np.array(img))
    print("已載入",i+1,"張")
#%%
Album = np.array(Album)
print("資料輸入完成")
#%%
df = pd.read_csv('D:/Anaconda3/程式碼/ML_ishs/預測交大女性對男性照片的喜好.csv')
Score = df['P1']
Score = np.array(Score)
Score = Score-1
print("標籤準備完成")
#%%
from sklearn.model_selection import train_test_split
X_train , X_test ,Y_train, Y_test = train_test_split(Album,Score,test_size=0.3,random_state=21,stratify=Score)
print("data分割完成")

X_train = X_train.reshape((646,400,400,1))
X_test = X_test.reshape((278,400,400,1))

from keras.utils import np_utils#1-hot encoding
Y_train = np_utils.to_categorical(Y_train,3)
Y_test = np_utils.to_categorical(Y_test,3)

#%%

from keras.models import Sequential
from keras.layers import Activation,Dense,Flatten,Dropout #flatten是要將矩陣拉平成向量
from keras.layers import Conv2D,MaxPooling2D #CNN與NN不同的地方在這
from keras.optimizers import SGD
import keras as kr

model = Sequential()#C-P-C-P-Dense-Dense

model.add(Conv2D(32,(7,7),padding='same',input_shape=(400,400,1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(4,4)))

model.add(Conv2D(64,(5,5),padding='same'))#filter的數目要越來越多
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3,3)))

model.add(Conv2D(128,(3,3),padding='same'))#filter的數目要越來越多
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(1000))
model.add(Activation('relu'))

model.add(Dense(3))
model.add(Activation('softmax'))

model.compile(optimizer='adam', 
              loss='categorical_crossentropy',
              metrics=['accuracy'])#組裝神經網路
#CNN到此已完成!
model.summary()
print("model建立完成")
#%%
model.fit(X_train,Y_train,batch_size=50,epochs= 3)
print("model訓練完成")
#%%
evalu = model.evaluate(X_test,Y_test) #結果測試-分數
print('loss=',evalu[0]) 
print('acc=',evalu[1])
#%%
model.save('D:/Anaconda3/程式碼/ML_ishs/man_model_924.h5')
print("model儲存完成")
#%%讀取原有model

model = tf.contrib.keras.models.load_model('D:/Anaconda3/程式碼/ML_ishs/man_model_924.h5')
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

#%%
predict = model.predict_classes(X_test)
#%%比較正確度
rad = np.random.randint(0,100)
print("預測為:",predict[rad])
print("實際為:",Y_test[rad])
image = X_test[rad].reshape(400,400)
plt.figure(num = 'haha' ,figsize=(6,6))
plt.imshow(image,cmap='gray')
