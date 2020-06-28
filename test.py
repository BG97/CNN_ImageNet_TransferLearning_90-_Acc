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
from keras.models import load_model
nb_classes = 10
img_channels = 3
img_rows = 112
img_cols = 112

if len(sys.argv)>1:
    train = sys.argv[1]
else:
    train = 'sub_imagenet/val'
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


X_test = np.array(data)
Y_test = np.array(labels)

model_file = sys.argv[2]
model = load_model(model_file)

Y_test = to_categorical(Y_test, nb_classes)

def score():
    score = model.evaluate(X_test,Y_test)
    print(model.metrics_names[1], score[1]*100)

score()
