import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten
from keras.layers import Conv2D,MaxPooling2D
import os

batch_size=32
num_classes=10
epochs=25
data_augmentation=True
num_prediction=20
save_dir=os.path.join(os.getcwd(),'saved_models')
model_name='keras_cifar10_trained_model.h5'

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print(x_train.shape)
print(x_train.shape[0])
print(x_test.shape[0])

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model=Sequential()
model.add(Conv2D(32,(3,3),activation='relu',padding='same',
                input_shape=x_train.shape[1:]))
model.add(Conv2D(32,(3,3),activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.25))

model.add(Conv2D(64,(3,3),activation='relu',padding='same',
                ))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes,activation='softmax'))
model.summary()
opt=keras.optimizers.rmsprop(lr=0.001,decay=1e-6)

model.compile(loss='categorical_crossentropy',
             optimizer=opt,
             metrics=['accuracy'])
x_train=x_train.astype('float32')
x_test=x_test.astype('float32')
x_train/=255
x_test/=255

if not data_augmentation:
    print('No using data augmentation')
    model.fit(x_train,y_train,
             batch_size=batch_size,
             epochs=epochs,
             validation_data=(x_test,y_test),
             shuffle=True)
else:
    print('Using real-time data augmentation')
    datagen=ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    rotation_range=0,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=False)
    
    datagen.fit(x_train)
    
    model.fit_generator(datagen.flow(x_train,y_train,
                                    batch_size=batch_size),
                       epochs=epochs,
                       validation_data=(x_test,y_test),workers=4)
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path=os.path.join(save_dir,model_name)
model.save(model_path)
print('Saved trained model at %s' % model_path)
scores=model.evaluate(x_test,y_test,verbose=1)
print('Test loss:',scores[0])
print('Test accuracy:',scores[1])
