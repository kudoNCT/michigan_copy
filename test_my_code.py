"""
Copyright (C) University of Science and Technology of China.
Licensed under the MIT License.
"""

import os
from collections import OrderedDict

import data
from options.test_options import TestOptions
from models.pix2pix_model import Pix2PixModel
from util.visualizer import Visualizer
from util.util import tensor2im, tensor2label, blend_image
from util import html
from data.base_dataset import single_inference_dataLoad,dataset_inference_dataLoad
from PIL import Image
import torch
import math
import numpy as np
import torch.nn as nn
import cv2
from glob import glob
import random
import os
import shutil




# read data
image_test = np.load("5k_image.npy")
path_dir = "../data_set/FFHQ/FFHQ/val_images/"
des_dir = "data_test_FFHQ/"
for i,name in enumerate(image_test):
    #name = name.replace('png','jpg')
    shutil.copy2(path_dir+name,des_dir)
    if i % 100 == 0:
        print(f'Processed:  {i} images')
