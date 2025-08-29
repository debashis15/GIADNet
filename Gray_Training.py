# import the libraries
import os
import numpy as np
import cv2
from conf import myConfig as config
from pathlib import Path
from conf import myConfig as config
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Layer
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import metrics
import tensorflow.data as tfdata
import tensorflow.image as tfimage
import tensorflow.nn as nn
import tensorflow.train as tftrain
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Conv2D,Activation,Input,Add,Subtract,AveragePooling2D,Multiply,Concatenate,Conv2DTranspose,BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import LearningRateScheduler
import tensorflow.keras.backend as K
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import LeakyReLU
from numpy import *
import random
import os
from glob import glob
import datetime
import argparse
import PIL
import tensorflow.keras.backend as K
from PIL import Image
from PIL import Image, ImageOps
os.environ["CUDA_VISIBLE_DEVICES"]='1'
# import the libraries
import os
import numpy as np
import cv2
#shape = (3, 3, 3, 1)
from pathlib import Path
import os
import numpy as np
import cv2
from conf import myConfig as config
from pathlib import Path
from conf import myConfig as config
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.layers import MaxPooling2D
import tensorflow as tf
from tensorflow.keras import models
import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import metrics
import tensorflow.data as tfdata
import tensorflow.image as tfimage
import tensorflow.nn as nn
import tensorflow.train as tftrain
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, Activation, Input, Add, Subtract, AveragePooling2D, Multiply, Concatenate, Conv2DTranspose, BatchNormalization
from tensorflow.keras.callbacks import LearningRateScheduler
import tensorflow.keras.backend as K
import random
import datetime
from PIL import Image

def kernel_hgrad(shape, dtype=None):
	kernel = np.zeros(shape)
	kernel[:,:,0,0] = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
	return kernel

def kernel_vgrad(shape, dtype=None):
	kernel = np.zeros(shape)
	kernel[:,:,0,0] = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
	return kernel

def gradient_extraction_block(input):


	L1_1 = Conv2D(1, (3, 3), strides=1, kernel_initializer=kernel_hgrad, use_bias=False, padding="same", trainable = False)(input)
	L1_2 = Conv2D(1, (3, 3), strides=1, kernel_initializer=kernel_vgrad, use_bias=False, padding="same", trainable = False)(input)
	L1_3 = tf.math.square(L1_1)
	L1_4 = tf.math.square(L1_2)
	L1_5 = Add()([L1_3, L1_4])
	L1_6 = tf.math.sqrt(L1_5)

	return L1_6


 
def feature_block(inputs):

    x = Conv2D(64, (3, 3), padding="same")(inputs)
    x = LeakyReLU(alpha=0.01)(x)

    x = Conv2D(64, (3, 3), dilation_rate=3, padding="same")(x)
    x = LeakyReLU(alpha=0.01)(x)

    x = Conv2D(64, (3, 3), dilation_rate=2, padding="same")(x)
    x = LeakyReLU(alpha=0.01)(x)

    x = Conv2D(64, (3, 3), dilation_rate=2, padding="same")(x)
    x = LeakyReLU(alpha=0.01)(x)

    x = Conv2D(64, (3, 3), dilation_rate=3, padding="same")(x)
    x = LeakyReLU(alpha=0.01)(x)

    x=Add()([inputs,x])

    #x = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x)

    x = Conv2D(64, (3, 3), padding="same")(x)
    x = LeakyReLU(alpha=0.01)(x)

    return x

def channel_attention_block(inputs):
    x1 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(inputs)
    x1 = Conv2D(64, (1, 1), strides=(1, 1), padding="same")(x1)
    x1 = LeakyReLU(alpha=0.01)(x1)
    x1 = Conv2D(64, (1, 1), strides=(1, 1), padding="same")(x1)
    x1 = LeakyReLU(alpha=0.01)(x1)

    x2 = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(inputs)
    x2 = Conv2D(64, (1, 1), strides=(1, 1), padding="same")(x2)
    x2 = LeakyReLU(alpha=0.01)(x2)
    x2 = Conv2D(64, (1, 1), strides=(1, 1), padding="same")(x2)
    x2 = LeakyReLU(alpha=0.01)(x2)

    x=Add()([x1,x2])

    x = Conv2D(64, (3, 3), strides=(1, 1), padding="same")(x)
    x = Activation('sigmoid')(x)
    x = Multiply()([inputs, x])

    return x

def multi_scale_block1(inputs):

    x = Conv2D(64, (3, 3),padding="same")(inputs)
    x = LeakyReLU(alpha=0.01)(x)
    x = Conv2D(64, (3, 3),padding="same")(x)
    x = LeakyReLU(alpha=0.01)(x)

    x=Add()([x,inputs])

    x = Conv2D(64, (3, 3),padding="same")(x)
    x = LeakyReLU(alpha=0.01)(x)
    x = Conv2D(64, (3, 3), padding="same")(x)
    x = Activation('sigmoid')(x)

    x = Multiply()([inputs, x])

    return x

def multi_scale_block2(inputs):

    x = Conv2D(64, (5, 5),padding="same")(inputs)
    x =LeakyReLU(alpha=0.01)(x)
    x = Conv2D(64, (5, 5), padding="same")(x)
    x = LeakyReLU(alpha=0.01)(x)

    x=Add()([x,inputs])

    x = Conv2D(64, (5, 5), padding="same")(x)
    x = LeakyReLU(alpha=0.01)(x)
    x = Conv2D(64, (5, 5), padding="same")(x)
    x = Activation('sigmoid')(x)

    x = Multiply()([inputs, x])

    return x

def multi_scale_block3(inputs):

    x = Conv2D(64, (4,4), padding="same")(inputs)
    x = LeakyReLU(alpha=0.01)(x)
    x = Conv2D(64, (4,4), padding="same")(x)
    x = LeakyReLU(alpha=0.01)(x)

    x=Add()([x,inputs])

    x = Conv2D(64, (4,4), padding="same")(x)
    x = LeakyReLU(alpha=0.01)(x)
    x = Conv2D(64, (4,4), padding="same")(x)
    x = Activation('sigmoid')(x)

    x = Multiply()([inputs, x])

    return x

def multi_scale_block4(inputs):

    
    x = Conv2D(64, (2,2),padding="same")(inputs)
    x =LeakyReLU(alpha=0.01)(x)
    x = Conv2D(64, (2,2),padding="same")(x)
    x =LeakyReLU(alpha=0.01)(x)

    x=Add()([x,inputs])

    x = Conv2D(64, (2,2),padding="same")(x)
    x =LeakyReLU(alpha=0.01)(x)
    x = Conv2D(64, (2,2),padding="same")(x)
    x = Activation('sigmoid')(x)

    x = Multiply()([inputs, x])

    return x


def denoise_net(input_shape=(None, None, 1)):
    input_img = tf.keras.layers.Input(shape=input_shape)

    z=gradient_extraction_block(input_img)
    y = Conv2D(31, (3,3),padding="same")(input_img)
    y = LeakyReLU(alpha=0.01)(y)

    x = Conv2D(32, (3,3),padding="same")(input_img)
    x = LeakyReLU(alpha=0.01)(x)
    
    z=Concatenate()([y,z])

    x4=input_img
    x4 = Conv2D(64, (3,3),dilation_rate=1,padding="same")(x4)
    x4 = LeakyReLU(alpha=0.01)(x4)
    x4 = Conv2D(64, (3,3),dilation_rate=2,padding="same")(x4)
    x4 = LeakyReLU(alpha=0.01)(x4)
    x4 = Conv2D(64, (3,3),dilation_rate=1,padding="same")(x4)
    x4 = LeakyReLU(alpha=0.01)(x4)
    
    x=Concatenate()([x,z])

    x=feature_block(x)
    x1=x
    x=feature_block(x)
    x2=x
    x=feature_block(x)
    x3=x
    x=feature_block(x)

    x=Add()([x,x4])

    x=Concatenate()([x,x1,x2,x3])

    x = Conv2D(64, (3, 3),padding="same")(x)
    x = LeakyReLU(alpha=0.01)(x)

    #x = Conv2D(32, (3, 3),padding="same")(x)
    #x = LeakyReLU(alpha=0.01)(x)

    x = channel_attention_block(x)

    y1 = multi_scale_block1(x)
    y2 = multi_scale_block2(x)
    y3 = multi_scale_block3(x)
    y4 = multi_scale_block4(x)

    x = Concatenate()([y1,y2,y3,y4])

    #x = Conv2D(128, (3, 3), padding="same")(x)
    #x = LeakyReLU(alpha=0.01)(x)

    x = Conv2D(64, (3, 3), padding="same")(x)
    x = LeakyReLU(alpha=0.01)(x)
    
    x = Concatenate()([x,z])

    x = Conv2D(32, (3, 3),padding="same")(x)
    x = LeakyReLU(alpha=0.01)(x)

    x = Conv2D(1,(3, 3),padding="same")(x)

    out=x
    model=Model(inputs=input_img, outputs=out)
    return model

# create custom learning rate scheduler
def lr_decay(epoch):
    initAlpha=0.001
    factor=0.5
    dropEvery=5
    alpha=initAlpha*(factor ** np.floor((1+epoch)/dropEvery))
    return float(alpha)
callbacks=[LearningRateScheduler(lr_decay)]

# create custom loss, compile the model
print("[INFO] compilingTheModel")
model = denoise_net(input_shape=(None, None, 1))
opt=optimizers.Adam(learning_rate=0.001)
def custom_loss(y_true,y_pred):
    diff=y_true-y_pred
    res=K.sum(diff*diff)/(2*config.batch_size)
    return res
model.compile(loss=custom_loss, optimizer=opt)
model.summary()

# load the data and normalize it
cleanImages=np.load(config.data)
print(cleanImages.dtype)
cleanImages=cleanImages/255.0
cleanImages=cleanImages.astype('float32')

# define augmentor and create custom flow
aug = ImageDataGenerator(rotation_range=30, fill_mode="nearest", validation_split=0.2)


def myFlow(generator,X):
    for batch in generator.flow(x=X,batch_size=config.batch_size,seed=0):
        noise=random.randint(0,61)
        trueNoiseBatch=np.random.normal(0,noise/255.0,batch.shape)
        noisyImagesBatch=batch+trueNoiseBatch
        yield(noisyImagesBatch,batch)



cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join('./training_checkpoints','ckpt_{epoch:03d}'), verbose=1,save_freq='epoch')
logdir = "./training_logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
lr_callback = [LearningRateScheduler(lr_decay)]
# train
model.fit_generator(myFlow(aug,cleanImages),
epochs=config.epochs,steps_per_epoch=len(cleanImages)//config.batch_size,callbacks=[lr_callback, cp_callback, tensorboard_callback], verbose=1)


