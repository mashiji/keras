from keras.models import Sequential
from keras.layers import Dense,Activation
model=Sequential()
model.add(Dense(32,activation='relu',input_dim=100))
model.add(Dense(1,activation='sigmoid'))
model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])
import numpy as np
data=np.random.random((1000,100))
labels=np.random.randint(2,size=(1000,1))
model.fit(data,labels,epochs=10,batch_size=32)


import keras
model=Sequential()
model.add(Dense(32,activation='relu',input_dim=100))
model.add(Dense(10,activation='softmax'))
model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
import numpy as np
data=np.random.random((1000,100))
labels=np.random.randint(10,size=(1000,1))
print(labels)
one_hot_labels=keras.utils.to_categorical(labels,num_classes=10)
print(one_hot_labels)
model.fit(data,one_hot_labels,epochs=10,batch_size=32)
