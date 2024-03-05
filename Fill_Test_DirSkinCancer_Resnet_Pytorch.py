# -*- coding: utf-8 -*-
"""

 Alfonso Blanco García , march 2024
"""

######################################################################
# PARAMETERS
######################################################################
#https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T
#Factor_split=0.995
From_dirname = "dataverse_files\\"
To_dirname = "Dir_Test_SkinCancer_Resnet_Pytorch\\"


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
data = pd.read_csv(From_dirname+'ISIC2018_Task3_Test_GroundTruth.csv')

TotImages=0


for i in range(len(TabDirName)):

    images = []
    NameImages=[]
    #TotTest=0
    

    for j in range(len(data)):
        #label = to_categorical(data['dx'][j], num_classes=7)
        label =data['dx'][j]
        if label !=TabDirName[i]: continue
        #img = load_img('HAM10000_images\\' + data['image_id'][j] + '.jpg', target_size=(224, 224))
        imgpath=From_dirname + 'ISIC2018_Task3_Test_Images\\ISIC2018_Task3_Test_Images\\' + data['image_id'][j] + '.jpg'
        #print(imgpath)
        img = cv2.imread(imgpath)
        if img is None: continue
        img=cv2.resize(img, (224,224))
        images.append(img)
        NameImages.append(data['image_id'][j] + '.jpg')
        TotImages+=1
        

    
    for k in range(len(images)):
        cv2.imwrite(To_dirname +"\\test\\" + TabDirName[i] + "\\" + NameImages[k], images[k])
        #TotImagesTest+=1
    print( TabDirName[i] + " has " + str(len(images)) + ", in test " ) 
   
print("Total images = " + str(TotImages))       

