# SkinLesionDetection_Resnet_Pytorch
Detection of skin lesions (among 7 classes) using the file https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T and using the pytorch resnet model. The success rate for the specific test file that comes with the download file is 81.13%.

According to the specifications of the download file, the 7 types of injuries to be detected are:

akiec : Actinic keratoses and intraepithelial carcinoma / Bowen’s disease

bkl : benign keratosis-like lesions (solar lentigines / seborrheic keratoses and lichen-planus like keratoses

bcc: basal cell carcinoma

df: dermatofibroma

mel: melanoma

nv: melanocytic nevi

vasc: vascular lesions (angiomas, angiokeratomas, pyogenic granulomas and hemorrhage


All packages, if any are missing, can be installed with a simple pip in case the programs indicate their absence in the environment.

Download all the files that accompany this project in a single folder.

By downloading the file from https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T in the directory where the project is located, a file called dataverse_files.zip is obtained, which once decompressed as dataverse_files contains, among others, the files HAM10000_images_part1.zip and HAM10000_images_part2.zip, which once unzipped must be unified into a single HAM10000_images folder (through a simple copy and paste) in the same dataverse_files directory

Next, the structure necessary for the operation of resnet pytorch is created, consisting of a folder Dir_SkinCancer_Resnet_Pytorch from which a folder called train and another called valid hang, each with a subfolder for each of the 7 classes, by executing:

Create_DirSkinCancer_Resnet_Pytorch.py

This structure is then filled from the images contained in dataverse_files\HAM10000_images and following the order indicated in the file dataverse_files\HAM10000_metadata, by executing:

Fill_DirSkinCancer_Resnet_Pytorch.py

To avoid resnet errors if you find a valid folder in which one of its subfolders does not have images: unzip the attached valid.zip and copy the resulting valid folder (be careful there may be two nested valid ones, only consider the last one) over the folder Dir_SkinCancer_Resnet_Pytorch overwriting the old one, this way you ensure that all valid subfolders have at least some image.

The model is trained and obtained by executing:

Train224x224_SkinCancer_Resnet_Pytorch.py

The execution log is attached as a file LOG_TrainSkinCancer_10epoch.txt and as a result the model checkpoint_SkinCancer_10epoch.pth is obtained (it is not attached because its size exceeds the file size limit that can be uploaded to github)

Next, the model obtained is tested: checkpoint_SkinCancer_10epoch.pth with the data from the specific test file that is attached to the download as ISIC2018_Task3_Test_Images.zip (it must be unzipped)

Next, the structure necessary for the operation of resnet pytorch is created with the specific test file, consisting of a folder Dir_Test_SkinCancer_Resnet_Pytorch from which a folder called test hangs with subfolders for each of the 7 classes, by executing:

Create_Test_DirSkinCancer_Resnet_Pytorch.py

This structure is then filled from the images contained in dataverse_files\ISIC2018_Task3_Test_Images\ISIC2018_Task3_Test_Images and following the order indicated in the file dataverse_files\ISIC2018_Task3_Test_GroundTruth.csv, by executing:

Fill_Test_DirSkinCancer_Resnet_Pytorch.py

The program is then executed:

Guess_Test_224x224SkinCancer_Resnet_Pytorch.py

Who performs the test

The log of its execution is attached as LOG_Test_SkinCancer.txt

The screen indicates the successes and failures, giving a success rate of 81.13%

To obtain predictions on images for which the skin lesion classification is not known and which are assumed to be in a folder called Test within the project, the program will be executed:

Recognize_SkinCancer_Resnet_Pytorch.py

Through the console, the prediction is obtained for each image and also in the output file

ModelsResults.txt

References:

https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T

https://medium.com/@lfoster49203/skin-lesion-classification-with-deep-learning-a-transfer-learning-approach-e1bc7d2b3d45

https://www.kaggle.com/code/hadeerismail/skin-cancer-prediction-cnn-acc-98

https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000
