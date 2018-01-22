#1
from keras.models import Sequential
from keras.layers import Dense, Activation
import keras
model=Sequential()
model.add(Dense(32,activation='relu',input_dim=100))
model.add(Dense(10,activation='softmax'))
model.compile(optimizer='rmsprop',
             loss='categorical_crossentropy',
             metrics=['accuracy'])
import numpy as np
data=np.random.random((1000,100))
labels=np.random.randint(10,size=(1000,1))
one_hot_labels=keras.utils.to_categorical(labels,num_classes=10)
model.fit(data,one_hot_labels,epochs=10,batch_size=32)



#2
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation
from keras.optimizers import SGD

import numpy as np
x_train=np.random.random((1000,20))
y_train=keras.utils.to_categorical(np.random.randint(10,size=(1000,1)),num_classes=10)
x_test=np.random.random((100,20))
y_test=keras.utils.to_categorical(np.random.randint(10,size=(100,1)),num_classes=10)
model=Sequential()
model.add(Dense(64,activation='relu',input_dim=20))
model.add(Dropout(0.5))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10,activation='softmax'))
sgd=SGD(lr=0.01,decay=1e-6,momentum=0.9,nesterov=True)
model.compile(loss='categorical_crossentropy',
             optimizer=sgd,metrics=['accuracy'])
model.fit(x_train,y_train,epochs=20,batch_size=128)
score=model.evaluate(x_test,y_test,batch_size=8)
print(score)

#3
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Dropout
x_train=np.random.random((1000,20))
y_train=np.random.randint(2,size=(1000,1))
x_test=np.random.random((100,20))
y_test=np.random.randint(2,size=(100,1))
model=Sequential()
model.add(Dense(64,input_dim=20,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',
             optimizer='rmsprop',
             metrics=['accuracy'])
model.fit(x_train,y_train,epochs=20,batch_size=128)
score=model.evaluate(x_test,y_test,batch_size=8)

#4
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten
from keras.layers import Conv2D,MaxPooling2D
from keras.optimizers import SGD
x_train=np.random.random((100,100,100,3))
y_train=keras.utils.to_categorical(np.random.randint(10,size=(100,1)),num_classes=10)
x_test=np.random.random((20,100,100,3))
y_test=keras.utils.to_categorical(np.random.randint(10,size=(20,1)),num_classes=10)

model=Sequential()
model.add(Conv2D(32,(3,3),activation='relu',input_shape=(100,100,3)))
model.add(Conv2D(32,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(64,(3,3),activation='relu'))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10,activation='softmax'))
sgd=SGD(lr=0.01,decay=1e-6,momentum=0.9,nesterov=True)
model.compile(loss='categorical_crossentropy',optimizer=sgd)
model.fit(x_train,y_train,batch_size=32,epochs=2)
model.summary()
score=model.evaluate(x_test,y_test,batch_size=32)
