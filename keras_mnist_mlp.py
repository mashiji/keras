from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense,Dropout

batch_size=128
num_classes=10
epochs=20

(x_train,y_train),(x_test,y_test)=mnist.load_data()
print(x_train.shape)
x_train=x_train.reshape(60000,784)
x_test=x_test.reshape(10000,784)
x_train=x_train.astype('float32')
x_test=x_test.astype('float32')
x_train/=255
x_test/=255
y_train=keras.utils.to_categorical(y_train,num_classes)
y_test=keras.utils.to_categorical(y_test,num_classes)
print(x_train.shape,'train samples')
print(x_test.shape,'test samples')
print(y_test.shape)

model=Sequential()
model.add(Dense(512,activation='relu',input_shape=(784,)))
model.add(Dropout(0.2))
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes,activation='softmax'))

model.summary()
model.compile(loss='categorical_crossentropy',
             optimizer='RMSprop',
             metrics=['accuracy'])
'''
训练模型
batch_size：指定梯度下降时每个batch包含的样本数
nb_epoch：训练的轮数，nb指number of
verbose：日志显示，0为不在标准输出流输出日志信息，1为输出进度条记录，2为epoch输出一行记录
validation_data：指定验证集
fit函数返回一个History的对象，其History.history属性记录了损失函数和其他指标的数值随epoch变化的情况，

如果有验证集的话，也包含了验证集的这些指标变化情况
'''
history=model.fit(x_train,y_train,
                 batch_size=batch_size,
                 epochs=epochs,
                 verbose=1,validation_data=(x_test,y_test))
score=model.evaluate(x_test,y_test,verbose=0)
print(score)
print('Test loss:',score[0])
print('Test accuracy:',score[1])
