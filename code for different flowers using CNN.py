from numpy import *
from pandas import *
from matplotlib.pyplot import *
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
model=Sequential()
model.add(Convolution2D(32,(3,3),input_shape=(64,64,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(init='uniform',activation='relu',output_dim=120))
model.add(Dense(init='uniform',activation='softmax',output_dim=5))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
from keras.preprocessing.image import  ImageDataGenerator
train_datagen=ImageDataGenerator(rescale=1./255,
                                 shear_range=0.2,
                                 zoom_range=0.2,
                                 horizontal_flip=True)
test_datagen =ImageDataGenerator(rescale=1./255)
x_train=train_datagen.flow_from_directory('flowers/training_set',
                                         target_size=(64,64),batch_size=32,class_mode='categorical')
x_test=train_datagen.flow_from_directory('flowers/test_set',
                                         target_size=(64,64),batch_size=32,class_mode='categorical')


model.fit_generator(x_train,steps_per_epoch=250,epochs=25,validation_data=x_test,validation_steps=63)
model.save("agrimodel.h5")

