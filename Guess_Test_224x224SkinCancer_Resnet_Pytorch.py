# -*- coding: utf-8 -*-
"""

 Alfonso Blanco García , Mar 2024
"""

######################################################################
# PARAMETERS
######################################################################
# https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T
######################################################################

TabSkinCancerComplete =[
            'akiec : Actinic keratoses and intraepithelial carcinoma / Bowen s disease', 
            'bkl : benign keratosis-like lesions (solar lentigines / seborrheic keratoses and lichen-planus like keratoses',
            'bcc : basal cell carcinoma',
            'df : dermatofibroma',
            'mel : melanoma', 
            'nv : melanocytic nevi',
            'vasc : vascular lesions (angiomas, angiokeratomas, pyogenic granulomas and hemorrhage,'    
           
           ]
TabSkinCancer =[ 'akiec',
            'bcc'  ,
            'bkl',      
            'df',       
            'mel',      
            'nv',    
            'vasc']
import torch
from torch import nn
import os
import re

import cv2

import numpy as np
import keras
import functools  
import time
inicio=time.time()

from torchvision import datasets, transforms, models
import torchvision.models as models
from PIL import Image

#model = models.resnet34(pretrained=True)
model = models.resnet50(pretrained=True)

# https://stackoverflow.com/questions/53612835/size-mismatch-for-fc-bias-and-fc-weight-in-pytorch
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 7)

#TabCarBrand=[]
def load_checkpoint(filepath):

    checkpoint = torch.load(filepath)
    
    #model.load_state_dict(checkpoint['state_dict'])
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    #model.class_to_idx = checkpoint['class_to_idx']
    
    return model

#model_path= "my_checkpoint1.pth"
model_path= "checkpoint_SkinCancer_10epoch.pth"


model = load_checkpoint('checkpoint_SkinCancer_10epoch.pth')
# Checking model i.e. should have 43 output units in the classifier
#print(model)
#DataPath='C:\\archiveKaggle\\cars_train\\cars_train' + '\\'

def find_classes(dir):
    classes = os.listdir(dir)
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx
#classes, c_to_idx = find_classes(data_dir+"/train")
classes, c_to_idx = find_classes('Dir_Test_SkinCancer_Resnet_Pytorch/test')


print(classes, c_to_idx) 


def loadimagesTest():
    imgpath = "Dir_Test_SkinCancer_Resnet_Pytorch\\test\\"

    print("Reading imagenes from ",imgpath)

    TabDirName=[]
    for root, dirnames, filenames in os.walk(imgpath):
         for dirname in dirnames:  
             print(dirname)
             #if len(dirname) < 2:
             #    dirnameTo= "0" + dirname
             #else:
             #     dirnameTo=  dirname    
             TabDirName.append(dirname)

    TotImages=0
   
    TotImagesValid=0
    TabImagePath = []
    NameImages=[]
    Y=[]
    
    for i in range(len(TabDirName)):
    
        imgpath1=imgpath+ str(TabDirName[i])+"\\"
        #print(imgpath1)
        
        # https://stackoverflow.com/questions/62137343/how-to-get-full-path-with-os-walk-function-in-python
        for root, dirnames, filenames in os.walk(imgpath1):
            for filename in filenames:  
                #print(filename)
                #if re.search("\.(jpg|jpeg|png|bmp|tiff)$", filename):
            
                filepath = os.path.join(root, filename)
                # https://stackoverflow.com/questions/51810407/convert-image-into-1d-array-in-python
            
                #image = cv2.imread(filepath)
                #cv2.imshow("image",image)
                #cv2.waitKey(0)
                #images.append(image)
                TabImagePath.append(filepath)
                NameImages.append(filename)
                Y.append(TabDirName[i])       
                TotImages+=1
                
                
    print( " Total images to test "  + str(TotImages))     

    return TabImagePath, Y, NameImages


def process_image(image):
    
    # Process a PIL image for use in a PyTorch model
  
    # Converting image to PIL image using image file path
    pil_im = Image.open(f'{image}')

    """

    # Building image transform
   
    """
    transform = transforms.Compose([transforms.Resize((224,224)),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


    # Transforming image for use with network
    pil_tfd = transform(pil_im)
    
    # Converting to Numpy array 
    array_im_tfd = np.array(pil_tfd)
    
    return array_im_tfd

def predict(image_path, model, topk=5):
    # Implement the code to predict the class from an image file   
    
    # Loading model - using .cpu() for working with CPUs
    loaded_model = load_checkpoint(model).cpu()
    # Pre-processing image
    img = process_image(image_path)
    # Converting to torch tensor from Numpy array
    img_tensor = torch.from_numpy(img).type(torch.FloatTensor)
    # Adding dimension to image to comply with (B x C x W x H) input of model
    img_add_dim = img_tensor.unsqueeze_(0)

    # Setting model to evaluation mode and turning off gradients
    loaded_model.eval()
    with torch.no_grad():
        # Running image through network
        output = loaded_model.forward(img_add_dim)
        
    #conf, predicted = torch.max(output.data, 1)   
    probs_top = output.topk(topk)[0]
    predicted_top = output.topk(topk)[1]
    
    # Converting probabilities and outputs to lists
    conf = np.array(probs_top)[0]
    predicted = np.array(predicted_top)[0]
        
    #return probs_top_list, index_top_list
    return conf, predicted


    
###########################################################
# MAIN
##########################################################

from tensorflow.keras.models import load_model


#TabImagePath_test, Y_test, imageName_test=loadimagesTest()
TabImagePath, Y_test, imageName_test=loadimagesTest()
#print(TabImagePath)
#print(Y_test)

TotalHits=0
TotalFailures=0
with open( "ModelsResults.txt" ,"w") as  w:
    
    
    
    for i in range(len(TabImagePath)):
        
                
        TabP=[]
        TabModel=[]
        TabPredictions1=[]
        

        conf, predicted1=predict(TabImagePath[i], model_path, topk=5)
        NameSkinCancerPredicted=TabSkinCancer[predicted1[0]]
        #NameSkinCancerTrue=TabSkinCancer[int(Y_test[i])-1]
        NameSkinCancerTrue=Y_test[i]
        #IntClassPredicted= predicted1[0]
        #ClassPredicted=str(IntClassPredicted)
        #if len(ClassPredicted) < 2 : ClassPredicted="0"+ClassPredicted
        
        if NameSkinCancerPredicted!=NameSkinCancerTrue:
            TotalFailures=TotalFailures + 1
            print("ERROR " + imageName_test[i]+ " is assigned Model " +  NameSkinCancerPredicted
                  + "  True Model " +  NameSkinCancerTrue)
                  
        else:
            print("HIT " + imageName_test[i]+ " is assigned model " +   NameSkinCancerPredicted)
                
          
                 
          
            TotalHits=TotalHits+1
        lineaw=[]
        lineaw.append(imageName_test[i]) 
        lineaw.append(Y_test[i])
        lineaw.append(NameSkinCancerTrue)
        lineaw.append(NameSkinCancerPredicted)
        lineaWrite =','.join(lineaw)
        lineaWrite=lineaWrite + "\n"
        w.write(lineaWrite)

    
print("")
print("Total hits = " + str(TotalHits))  
print("Total failures = " + str(TotalFailures) )     
print("Accuracy = " + str(TotalHits*100/(TotalHits + TotalFailures)) + "%") 
