# -*- coding: utf-8 -*-
"""

 Alfonso Blanco García , march 2024
"""

######################################################################
# PARAMETERS
######################################################################
#https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T
Factor_split=0.995
From_dirname = "dataverse_files\\"
To_dirname = "Dir_SkinCancer_Resnet_Pytorch\\"


######################################################################

import os
import re

import cv2

import numpy as np

#imgpath = From_dirname + "\\"

#print("Reading imagenes from ",imgpath)

TabDirName=[]
TabDirName.append("bkl")
TabDirName.append("df")
TabDirName.append("nv")
TabDirName.append("mel")
TabDirName.append("vasc")
TabDirName.append("bcc")
TabDirName.append("akiec")


# https://medium.com/@lfoster49203/skin-lesion-classification-with-deep-learning-a-transfer-learning-approach-e1bc7d2b3d45
import pandas as pd
from keras.preprocessing.image import load_img, img_to_array
from keras.utils import to_categorical
data = pd.read_csv(From_dirname+'HAM10000_metadata')

TotImages=0


for i in range(len(TabDirName)):

    images = []
    NameImages=[]
    TotImagesTrain=0
    TotImagesValid=0

    for j in range(len(data)):
        #label = to_categorical(data['dx'][j], num_classes=7)
        label =data['dx'][j]
        if label !=TabDirName[i]: continue
        #img = load_img('HAM10000_images\\' + data['image_id'][j] + '.jpg', target_size=(224, 224))
        img = cv2.imread(From_dirname + 'HAM10000_images\\' + data['image_id'][j] + '.jpg')
        img=cv2.resize(img, (224,224))
        images.append(img)
        NameImages.append(data['image_id'][j] + '.jpg')
        TotImages+=1
        #imgpath1=imgpath+ str(TabDirName[i])+"\\"

    limit=int(len(images)*Factor_split)
    for k in range(len(images)):
        if k > limit:
            cv2.imwrite(To_dirname +"\\valid\\" + TabDirName[i] + "\\" + NameImages[k], images[k])
            TotImagesValid+=1
        else:    
            cv2.imwrite(To_dirname +"\\train\\" +TabDirName[i] + "\\" + NameImages[k], images[k])
            TotImagesTrain+=1
    print( TabDirName[i] + " has " + str(len(images)) + ", in train " + str(TotImagesTrain)  +  " in valid " + str(TotImagesValid)) 
   
print("Total images = " + str(TotImages))       

