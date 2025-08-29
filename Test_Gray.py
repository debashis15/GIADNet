#importLibraries
import os.path
import tensorflow as tf
from tensorflow.keras.models import load_model
from conf import myConfig as config
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import argparse
from pathlib import Path
import tensorflow.keras.backend as K
from matplotlib import pyplot as plt
import scipy.io
from scipy import ndimage
from tifffile import imwrite
from utils import utils_image as util
from PIL import Image
import skimage.feature
from skimage.util import random_noise

os.environ["CUDA_VISIBLE_DEVICES"]="2"

# custom filter
def my_Hfilter(shape, dtype=None):

    f = np.array([
            [[[-1]], [[0]], [[1]]],
            [[[-2]], [[0]], [[2]]],
            [[[-1]], [[0]], [[1]]]
    ], dtype=np.float32)
    assert f.shape == shape
    return K.variable(f, dtype='float32')

def my_Vfilter(shape, dtype=None):

    f = np.array([
            [[[-1]], [[-2]], [[-1]]],
            [[[0]], [[0]], [[0]]],
            [[[1]], [[2]], [[1]]]
    ], dtype=np.float32)
    assert f.shape == shape
    return K.variable(f, dtype='float32')

#ParsingArguments
parser=argparse.ArgumentParser()
parser.add_argument('--dataPath',dest='dataPath',type=str,default='./BSD68',help='testDataPath')
parser.add_argument('--weightsPath',dest='weightsPath',type=str,default='./training_checkpoints/ckpt_{0:03d}',help='pathOfTrainedCNN')
parser.add_argument('--epoch',dest='epoch',type=int,default=10)
parser.add_argument('--noise',dest='noise_level_img',type=int,default=10)
args=parser.parse_args()
#createModel, loadWeights
def custom_loss(y_true,y_pred): #this is required for loading a keras-model created with custom-loss
    diff=K.abs(y_true-y_pred)
    res=(diff)/(config.batch_size)
    return res

nmodel_PROPOSED=load_model(args.weightsPath,custom_objects={'my_Hfilter': my_Hfilter,'my_Vfilter': my_Vfilter,'custom_loss':custom_loss})
print('Trained Model is loaded')

#createArrayOfTestImages
p=Path(args.dataPath)
listPaths=list(p.glob('./*.png'))
imgTestArray = []
for path in listPaths:
    imgTestArray.append((
    (cv2.imread(str(path),0))))
# imgTestArray=np.array(imgTestArray)/255.
imgTestArray = np.array(imgTestArray, dtype=object) / 255.

length=68

folder_path_original = "./Test_Results/Synthetic/"+str(args.epoch)+"/"+str(args.noise_level_img)+"/Original/"
if not os.path.exists(folder_path_original):
    os.makedirs(folder_path_original)
    
folder_path_noisy = "./Test_Results/Synthetic/"+str(args.epoch)+"/"+str(args.noise_level_img)+"/Noisy/"
if not os.path.exists(folder_path_noisy):
    os.makedirs(folder_path_noisy)

folder_path_proposed = "./Test_Results/Synthetic/"+str(args.epoch)+"/"+str(args.noise_level_img)+"/Proposed/"
if not os.path.exists(folder_path_proposed):
    os.makedirs(folder_path_proposed)
    
metric_path_proposed = "./Test_Results/Synthetic/"+str(args.epoch)+"/"+str(args.noise_level_img)+"/metric.txt"
file_m = open(metric_path_proposed,'w')
file_m.write('Metric: '+args.dataPath+'; Epoch: '+str(args.epoch)+'; Noise Level: '+str(args.noise_level_img)+'\n')
file_m.close()



sumPSNR=0
sumSSIM=0
psnr_val=np.empty(length)
ssim_val=np.empty(length)
for i in range(0,length):
    np.random.seed(seed=0)  # for reproducibility
    img1=imgTestArray[i]
    f=img1 + np.random.normal(0, args.noise_level_img/255., img1.shape)
    z=np.squeeze(nmodel_PROPOSED.predict(np.expand_dims(f,axis=0)))
    cv2.imwrite(folder_path_original+str(i+1)+"_Original.png",255.*img1)
    cv2.imwrite(folder_path_noisy+str(i+1)+"_Noisy.png",255.*f)
    cv2.imwrite(folder_path_proposed+str(i+1)+"_enhAGSDNet_Model2.png",255.*z)
    # psnr_val[i]=psnr(img1,z)
    # print(img1.dtype)
    psnr_val[i] = psnr(img1.astype(np.float32), z)
    # ssim_val[i]=ssim(img1,z)
    ssim_val[i] = ssim(img1, z, channel_axis=-1)
    file_m = open(metric_path_proposed,'a')
    file_m.write('PSNR of image '+str(i+1)+' is '+str(psnr_val[i])+'\n')
    file_m.write('SSIM of image '+str(i+1)+' is '+str(ssim_val[i])+'\n')
    file_m.close()
    sumPSNR=sumPSNR+psnr_val[i]
    sumSSIM=sumSSIM+ssim_val[i]
avgPSNR=sumPSNR/length
avgSSIM=sumSSIM/length
file_m = open(metric_path_proposed,'a')
file_m.write('\n\navgPSNR on dataset = '+str(avgPSNR)+'\n')
file_m.write('avgSSIM on dataset = '+str(avgSSIM)+'\n\n')
file_m.close()
