
# coding: utf-8

# In[22]:


import cv2
import numpy as np
import os
from random import shuffle
from tqdm import tqdm
TRAIN_DIR='/home/msj/dataset_kaggledogvscat/train/train/'
TEST_DTR='/home/msj/dataset_kaggledogvscat/test'
IMG_SIZE=224

def label_img(img):
    world_label=img.split('.')[0]
    if world_label=='cat':
        return [1,0]
    elif world_label=='dog':
        return [0,1]
    
def create_train_data():
    training_data=[]
    for img in tqdm(os.listdir(TRAIN_DIR)):
        label=label_img(img)
        path=os.path.join(TRAIN_DIR,img)
        img=cv2.imread(path,cv2.IMREAD_COLOR)
        img=cv2.resize(img,(IMG_SIZE,IMG_SIZE))
        training_data.append([np.array(img),np.array(label)])
    shuffle(training_data)
    np.save('/home/msj/train_data1.npy',training_data)
    return training_data

def process_test_data():
    testing_data=[]
    for img in tqdm(os.listdir(TEST_DTR)):
        path=os.path.join(TEST_DTR,img)
        img_num=img.split('.')[0]
        img=cv2.imread(path,cv.IMREAD_COLOR)
        img=cv2.resize(img,(IMG_SIZE,IMG_SIZE))
        testing_data.append([np.array(img),img_num])
    shuffle(test_data)
    np.save('/home/msj/test_data.npy',training_data)
    return testg_data
        
train_data=create_train_data()
train=train_data[:-1000]
test=train_data[-1000:]

X=np.array([i[0] for i in train]).reshape(-1,IMG_SIZE,IMG_SIZE,3)
Y=np.array([i[1] for i in train])

test_x=np.array([i[0] for i in test]).reshape(-1,IMG_SIZE,IMG_SIZE,3)
test_y=np.array([i[1] for i in test])



from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense,GlobalAveragePooling2D
from keras import backend as k
from matplotlib.pyplot import imshow
from keras.applications.imagenet_utils import preprocess_input
import numpy as np
import imageio 
from keras.utils.vis_utils import model_to_dot  
from keras.models import load_model


base_model=ResNet50(weights='imagenet',include_top=False)
x=base_model.output
x=GlobalAveragePooling2D()(x)
x=Dense(1024,activation='relu')(x)
predictions=Dense(2,activation='softmax')(x)

model=Model(inputs=base_model.input,outputs=predictions)

for layer in base_model.layers:
    layer.trainable=False

model.compile(optimizer='rmsprop',loss='binary_crossentropy',
             metrics=['accuracy'])
model.fit(X,Y,epochs=10,batch_size=64)
preds=model.evaluate(test_x,test_y)
print("Loss="+str(preds[0]))
print("Test Accuracy="+str(preds[1]))

model.save('/home/msj/cat_dog.h5')

model=load_model('/home/msj/cat_dog.h5')
img_path='/home/msj/dataset_kaggledogvscat/test/10.jpg'
img=image.load_img(img_path,target_size=(224,224))
x=image.img_to_array(img)
x=np.expand_dims(x,axis=0)
x=preprocess_input(x)
my_image=imageio.imread(img_path)
imshow(my_image)
result=model.predict(x)
model.summary()
if result[0][0]>result[0][1]:
    print('This is a cat!')
else:
    print('This is a dog!')

