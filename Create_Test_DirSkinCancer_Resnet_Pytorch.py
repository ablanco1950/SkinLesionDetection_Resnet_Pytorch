import os
import shutil
output_dir="Dir_Test_SkinCancer_Resnet_Pytorch"

if  os.path.exists(output_dir):shutil.rmtree(output_dir)
os.mkdir(output_dir)
os.mkdir(output_dir + "\\test")
os.mkdir(output_dir + "\\test\\bkl")
os.mkdir(output_dir + "\\test\\df")
os.mkdir(output_dir + "\\test\\nv")
os.mkdir(output_dir + "\\test\\mel")
os.mkdir(output_dir + "\\test\\vasc")
os.mkdir(output_dir + "\\test\\bcc")
os.mkdir(output_dir + "\\test\\akiec")

