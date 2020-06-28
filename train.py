from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Activation, Flatten
#from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D
#from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.utils import np_utils, generic_utils, to_categorical
from sklearn.utils import shuffle
import keras	
import sys
#from keras import regularizers
#from keras.regularizers import l2
import numpy as np
from keras import backend as K
#K.set_image_dim_ordering('th')
from keras.layers import Input
import os
from PIL import Image
from skimage.transform import resize
from keras.applications.vgg16 import VGG16
from keras.layers import GlobalAveragePooling2D
from keras import optimizers

nb_classes = 10
img_channels = 3
img_rows = 112
img_cols = 112

if len(sys.argv)>1:
    train = sys.argv[1]
else:
    train = 'sub_imagenet/train'
img_folders = os.listdir(train)
data = []
labels = []
for i in range(len(img_folders)):
    images = os.listdir(train+"/"+img_folders[i])
    for img in images:
        image = Image.open(train+"/"+img_folders[i]+"/"+img)
        image = np.array(image, dtype='uint8')
        data.append(resize(image, (224,224,3)))
        labels.append(i)
    

X_train = np.array(data)
Y_train = np.array(labels)
Y_train = to_categorical(Y_train, nb_classes)

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print('Y_train shape:', Y_train.shape)

X_train,Y_train = shuffle(X_train,Y_train)

image_input = Input(shape=(224,224,3))
model = VGG16(include_top=False, weights='imagenet', input_tensor=image_input, pooling=None,classes=nb_classes)

for layer in model.layers[:9]:
    layer.trainable = False


output = model.output
output = Flatten()(output)
output = Dense(1024, activation="relu")(output)
predictions = Dense(10, activation="softmax")(output)

 
model_final = Model(input = model.input, output = predictions)
#keras.utils.multi_gpu_model(model, gpus=2, cpu_merge=False, cpu_relocation=False)
#opt = keras.optimizers.Adamax(lr=0.001, beta_1=0.9, beta_2=0.99, decay=1e-6)# best one
opt = optimizers.SGD(lr=0.0001,momentum=0.9)

model_final.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])



def train():
    model_final.fit(X_train, Y_train,
              batch_size=32,
              epochs=20,
              shuffle=True)
    if len(sys.argv)>1:
        model_final.save(sys.argv[2])
    else:
        model_save('model.h5')
train()
