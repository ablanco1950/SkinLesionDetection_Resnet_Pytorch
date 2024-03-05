import os
import shutil
output_dir="Dir_SkinCancer_Resnet_Pytorch"

if  os.path.exists(output_dir):shutil.rmtree(output_dir)
os.mkdir(output_dir)
os.mkdir(output_dir + "\\train")
os.mkdir(output_dir+ "\\valid")
os.mkdir(output_dir + "\\train\\bkl")
os.mkdir(output_dir + "\\train\\df")
os.mkdir(output_dir + "\\train\\nv")
os.mkdir(output_dir + "\\train\\mel")
os.mkdir(output_dir + "\\train\\vasc")
os.mkdir(output_dir + "\\train\\bcc")
os.mkdir(output_dir + "\\train\\akiec")
os.mkdir(output_dir + "\\valid\\bkl")
os.mkdir(output_dir + "\\valid\\df")
os.mkdir(output_dir + "\\valid\\nv")
os.mkdir(output_dir + "\\valid\\mel")
os.mkdir(output_dir + "\\valid\\vasc")
os.mkdir(output_dir + "\\valid\\bcc")
os.mkdir(output_dir + "\\valid\\akiec")
