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
#%%

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Album = []
for i in range(10000):
    img = Image.open('D:/Anaconda3/程式碼/ML_ishot/'+np.str(i)+'.jpg').convert('L')
    #後面的convert可以將圖片轉成灰階
    img = img.resize((200, 200),Image.ANTIALIAS)
    #plt.figure(num = np.str(i) ,figsize=(10,10))
    #plt.imshow(img)
    img = np.array(img)
    img = img/255
    Album.append(np.array(img))
    print("已載入",i+1,"張")
    
print("資料輸入完成")

#%%  
Album = np.array(Album)
print("資料輸入完成")
plt.figure(num = 'test' ,figsize=(4,4))
plt.imshow(Album[0])
#%%
Score = np.random.randint(0,4,10000)
print("標籤準備完成")
#%%
#Album  = Album.reshape((1000,1000,1000,3))
from sklearn.model_selection import train_test_split
X_train , X_test ,Y_train, Y_test = train_test_split(Album,Score,test_size=0.3,random_state=1,stratify=Score)
print("data分割完成")
from keras.utils import np_utils#1-hot encoding
Y_train = np_utils.to_categorical(Y_train,4)
Y_test = np_utils.to_categorical(Y_test,4)
#%%
X_train =X_train.reshape(7000,200,200,1)
X_test =X_test.reshape(3000,200,200,1)
#%%
from keras.models import Sequential
from keras.layers import Activation,Dense,Flatten,Dropout #flatten是要將矩陣拉平成向量
from keras.layers import Conv2D,MaxPooling2D #CNN與NN不同的地方在這
from keras.optimizers import SGD
import keras as kr

model = Sequential()#C-P-C-P-Dense-Dense

model.add(Conv2D(32,(7,7),padding='same',input_shape=(200,200,1)))
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

model.add(Dense(4))
model.add(Activation('softmax'))

model.compile(optimizer='adam', 
              loss='categorical_crossentropy',
              metrics=['accuracy'])#組裝神經網路
#CNN到此已完成!
model.summary()
print("model建立完成")
#%%
model.fit(X_train,Y_train,batch_size=50,epochs=12)
print("model訓練完成")
evalu = model.evaluate(X_test,Y_test) #結果測試-分數
print('loss=',evalu[0]) 
print('acc=',evalu[1])
#%%
model.save('my_model_10000.h5')
print("model儲存完成")
#%%讀取原有model
model = tf.contrib.keras.models.load_model('my_model_10000.h5')
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

#%%
predict = model.predict_classes(X_test)
#%%比較正確度
rad = np.random.randint(0,3000)
print("預測為:",predict[rad])
print("實際為:",Y_test[rad])
image = X_test[rad].reshape(400,400)
plt.figure(num = 'haha' ,figsize=(8,8))
plt.imshow(image,cmap='gray')
