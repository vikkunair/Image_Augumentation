#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  7 07:31:37 2020

@author: vikasnair
"""

import pandas as pd
from math import floor
import random

test_proportion = 0.05
validation_proportion = 0.1
train_proportion = 1 - (test_proportion+validation_proportion)

total_files = 16185

test_no_files = floor(test_proportion*total_files)
val_no_files = floor(validation_proportion*total_files)
train_no_files = total_files - (test_no_files+val_no_files)


#transformed_images_folder = ""
#original_images_folder = ""

original_files_list_file  = '/Users/vikasnair/Documents/Personal/Surrey_MSc/Image_Processing_and_Deep_Learning/Amber_Download/cars/allimages.txt'


data = pd.read_csv(original_files_list_file, sep=" ", header=None)
data.columns = ["File_Full_Path", "Label"]

data['File_Name'] = ''


for j in range(0,len(data)):
    data['File_Name'][j] = data['File_Full_Path'][j].split('/')[-1] +'$'+ str(data['Label'][j])
    
ori_file_names = list(data['File_Name'])
    
test_images = []
val_images = []

for k in range(0,test_no_files):
    image_chosen = random.choice(ori_file_names)
    test_images.append(image_chosen)
    ori_file_names.remove(image_chosen)

for k in range(0,val_no_files):
    image_chosen = random.choice(ori_file_names)
    val_images.append(image_chosen)
    ori_file_names.remove(image_chosen)
    
train_images = ori_file_names

#Transformed Images
import glob
transformed_images_list = glob.glob("/Users/vikasnair/Documents/Personal/Surrey_MSc/Image_Processing_and_Deep_Learning/Amber_Download/Python_Augumenter/*.jpg")

transformed_images_df = pd.DataFrame(transformed_images_list)
transformed_images_df.columns = ["image_path"]

transformed_images_df["File_name"] = ""

ori_file_names_2 = list(data['File_Name'])




#Adding labels
for j in range(0,len(transformed_images_df)):
    transformed_images_df['File_name'][j] = transformed_images_df['image_path'][j].split('/')[-1]
    term = (transformed_images_df['File_name'][j].split('_')[0])
    
    for k in range(0,len(ori_file_names_2)):
        if term == (ori_file_names_2[k].split('.')[0]):
            transformed_images_df['File_name'][j] = transformed_images_df['File_name'][j] + '$' + ori_file_names_2[k].split('$')[1]     
    print(j)
    
transformed_images_df["Train_Flag"] = 0
transformed_images_df["Labels"] = ""
transformed_images_df["File_Name_ext"] = ""
ori_file_names_3 = [x.split('.')[0] for x in ori_file_names ]

for j in range(0,len(transformed_images_df)):
    transformed_images_df["Labels"][j] = transformed_images_df['File_name'][j].split("$")[1]
    transformed_images_df["File_Name_ext"][j] =transformed_images_df['File_name'][j].split("$")[0]
    
    for i in range(0,len(train_images)):
        if transformed_images_df["File_Name_ext"][j].split('_')[0] == train_images[i].split('.')[0]:
            transformed_images_df["Train_Flag"][j] = 1
    
    print(j)
    
    
#To run    
    
    
transformed_images_df.to_csv("/Users/vikasnair/Documents/Personal/Surrey_MSc/Image_Processing_and_Deep_Learning/Amber_Download/Trans_2_splits/transformed_with_labels_1.csv",index=False)

transformed_images_df = pd.read_csv("/Users/vikasnair/Documents/Personal/Surrey_MSc/Image_Processing_and_Deep_Learning/Amber_Download/Trans_2_splits/transformed_with_labels_1.csv")

training_final_df_combined = transformed_images_df[transformed_images_df["Train_Flag"]==1]
training_final_df_combined = training_final_df_combined.reset_index()
combine_frames = [training_final_df_combined["File_name"],pd.DataFrame(train_images)]
training_final_df_combined = pd.concat(combine_frames)


#Creating final txt files
#Train
output_folder = "/Users/vikasnair/Documents/Personal/Surrey_MSc/Image_Processing_and_Deep_Learning/Amber_Download/Trans_2_splits/85_05_10/"
vm_path = "path_to_vm/"


training_final_df_combined = training_final_df_combined.reset_index()
del training_final_df_combined["index"]
training_final_df_combined.columns = ["File_Name_Label"]
training_final_df_combined["File_Name"] = ""
training_final_df_combined["Label"] = ""

for j in range(0,len(training_final_df_combined)):
    training_final_df_combined["File_Name"][j] = vm_path+(training_final_df_combined["File_Name_Label"][j]).split("$")[0]
    training_final_df_combined["Label"][j] = (training_final_df_combined["File_Name_Label"][j]).split("$")[1]
    print(j)

training_final_df_combined = training_final_df_combined.loc[:,["File_Name","Label"]]
    
training_final_df_combined.to_csv(output_folder+"train.txt", header=None, index=None, sep=' ', mode='a')

    
#Testing
test_images = pd.DataFrame(test_images)
test_images = test_images.reset_index()
del test_images["index"]
test_images.columns = ["File_Name_Label"]
test_images["File_Name"] = ""
test_images["Label"] = ""

for j in range(0,len(test_images)):
    test_images["File_Name"][j] = vm_path+(test_images["File_Name_Label"][j]).split("$")[0]
    test_images["Label"][j] = (test_images["File_Name_Label"][j]).split("$")[1]
    print(j)

test_images = test_images.loc[:,["File_Name","Label"]]
    
    
test_images.to_csv(output_folder+"test.txt", header=None, index=None, sep=' ', mode='a')


#Validation
val_images = pd.DataFrame(val_images)
val_images = val_images.reset_index()
del val_images["index"]
val_images.columns = ["File_Name_Label"]
val_images["File_Name"] = ""
val_images["Label"] = ""

for j in range(0,len(val_images)):
    val_images["File_Name"][j] = vm_path+(val_images["File_Name_Label"][j]).split("$")[0]
    val_images["Label"][j] = (val_images["File_Name_Label"][j]).split("$")[1]
    print(j)

val_images = val_images.loc[:,["File_Name","Label"]]
    
    
val_images.to_csv(output_folder+"validation.txt", header=None, index=None, sep=' ', mode='a')


#AllImages
combine_frames = [pd.DataFrame(ori_file_names_2),pd.DataFrame((transformed_images_df["File_name"].to_list()))]
master_set = pd.concat(combine_frames)

master_set = master_set.reset_index()
del master_set["index"]
master_set.columns = ["File_Name_Label"]
master_set["File_Name"] = ""
master_set["Label"] = ""

for j in range(0,len(master_set)):
    master_set["File_Name"][j] = vm_path+(master_set["File_Name_Label"][j]).split("$")[0]
    master_set["Label"][j] = (master_set["File_Name_Label"][j]).split("$")[1]
    print(j)

master_set = master_set.loc[:,["File_Name","Label"]]
    
    
master_set.to_csv(output_folder+"allimages.txt", header=None, index=None, sep=' ', mode='a')
