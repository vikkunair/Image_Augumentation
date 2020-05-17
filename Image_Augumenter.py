#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 16 00:26:47 2020

@author: vikasnair
"""

#Image data augumentation

from glob import glob
from PIL import Image
import numpy as np
import random
import cv2
from skimage.transform import rotate,warp, AffineTransform
from skimage.util import random_noise
import matplotlib
source_path = "/Users/vikasnair/Documents/Personal/Surrey_MSc/Image_Processing_and_Deep_Learning/Amber_Download/cars/car_ims/"
destination_path = "/Users/vikasnair/Documents/Personal/Surrey_MSc/Image_Processing_and_Deep_Learning/Amber_Download/Python_Augumenter/"

file_list = glob(source_path+"*.jpg")


for i,e in enumerate(file_list):
    
    file_name_current = e.split('/')[-1]
    

    #Image Load
    image = Image.open(e)
    image_array = np.array(image)


    #Image Dimensions
    #height = image_array.shape[0]
    #weight = image_array.shape[1]
    
    #Image Flip Horizoantal
    flipped_img = np.fliplr(image)
    conv_im = Image.fromarray(flipped_img)
    conv_im.save(destination_path+"/"+file_name_current.split(".")[0]+'_'+'horif'+'_'+str(random.randint(0, 100))+".jpg")
    
    
    #Image Flip vertical
    flipped_img = np.flipud(image)
    conv_im = Image.fromarray(flipped_img)
    conv_im.save(destination_path+"/"+file_name_current.split(".")[0]+'_'+'vertif'+'_'+str(random.randint(0, 100))+".jpg")
    
    
    #Rotation Anticloackwise Rotation
    rotated_new_image = rotate(image_array,angle = random.randint(1, 360))
    matplotlib.image.imsave(destination_path+"/"+file_name_current.split(".")[0]+'_'+'rotanti'+'_'+str(random.randint(0, 100))+".jpg",rotated_new_image)
    #rotated_new_image.save(destination_path+"/"+file_name_current.split(".")[0]+'_'+'rotanti'+'_'+str(random.randint(0, 100))+".jpg")
    
    #Clockwise rotation
    rotated_new_image = rotate(image_array,angle = -1*(random.randint(1, 360)))
    matplotlib.image.imsave(destination_path+"/"+file_name_current.split(".")[0]+'_'+'rotclock'+'_'+str(random.randint(0, 100))+".jpg",rotated_new_image)
   
    
    #Add Noise
    noisy_img = random_noise(image_array, mode='s&p',amount=0.3)
    matplotlib.image.imsave(destination_path+"/"+file_name_current.split(".")[0]+'_'+'noisy'+'_'+str(random.randint(0, 100))+".jpg",noisy_img)
    
    #Blurring the image
    kernal_size = random.randrange(1,12,2)
    blurred_image = cv2.GaussianBlur(image_array,(kernal_size,kernal_size),0)
    matplotlib.image.imsave(destination_path+"/"+file_name_current.split(".")[0]+'_'+'blur'+'_'+str(random.randint(0, 100))+".jpg",blurred_image)
    
    
    #Shifting
    transform = AffineTransform(translation=(random.randint(10, 100),random.randint(10, 100)))
    warped_image = warp(image_array,transform,mode = 'wrap')
    matplotlib.image.imsave(destination_path+"/"+file_name_current.split(".")[0]+'_'+'shift'+'_'+str(random.randint(0, 100))+".jpg",warped_image)

    print(i)
    