import tensorflow as tf
import numpy as np
from Parameters import initialParams as prm
from tensorflow.keras.layers import Input, Conv2D, LeakyReLU,BatchNormalization, Activation, Add, AveragePooling2D,MaxPooling2D,Concatenate,Subtract,Multiply,LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l1
import os
os.environ["CUDA_VISIBLE_DEVICES"]=prm.gpuid
from tensorflow.keras import backend as K




#------------------------------------------------------------------------------------------------------------------
#Gradient Extraction Block
#------------------------------------------------------------------------------------------------------------------

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

    x = LeakyReLU(alpha=0.01)(x)
    x=Add()([x,z])

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


