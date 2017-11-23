import numpy as np
from PIL import Image
from PIL import ImageEnhance
import matplotlib.pyplot as plt
import os
path_trian_dir='./train3'
path_train_save_dir='./train_new'
num=134
nums_total=7

def rmrf_mkdir(dirname):
    if os.path.exists(dirname):
        shutil.rmtree(dirname)
    os.mkdir(dirname)
    for i in range(num):
        os.mkdir(os.path.join(dirname,str(i)))
def color_light_change(image_data,color_ordering=0):
    reference=[0.8,1.2,1.5]
    index_bri=np.random.randint(3)
    brightness=reference[index_bri]
    index_col=np.random.randint(3)
    color=reference[index_col]
    index_cont=np.random.randint(3)
    contrast=reference[index_cont]
    if color_ordering==0:
        enh_bri=ImageEnhance.Brightness(image_data)
        image_data=enh_bri.enhance(brightness)
        enh_col=ImageEnhance.Color(image_data)
        image_data=enh_col.enhance(color)
        enh_con=ImageEnhance.Contrast(image_data)
        image_data=enh_con.enhance(contrast)
    elif color_ordering==1:
        enh_col=ImageEnhance.Color(image_data)
        image_data=enh_col.enhance(color)
        enh_con=ImageEnhance.Contrast(image_data)
        image_data=enh_con.enhance(contrast)
        enh_bri=ImageEnhance.Brightness(image_data)
        image_data=enh_bri.enhance(brightness)
    elif color_ordering==2:
        enh_con=ImageEnhance.Contrast(image_data)
        image_data=enh_con.enhance(contrast)
        enh_bri=ImageEnhance.Brightness(image_data)
        image_data=enh_bri.enhance(brightness)
        enh_col=ImageEnhance.Color(image_data)
        image_data=enh_col.enhance(color)
    return image_data

def preprocess_image(image,convert_flag=0):
   
    brightness=1.5
    color=1.5
    contrast=1.5
    #if convert_flag==0:
    #    image=image.transpose(Image.FLIP_LEFT_RIGHT)
    if convert_flag==0:
        image=color_light_change(image,np.random.randint(3))
    elif convert_flag==1:
        image=image.transpose(Image.FLIP_LEFT_RIGHT)
        image=color_light_change(image,np.random.randint(3))
    elif convert_flag==2:
        enh_col=ImageEnhance.Color(image)
        image=enh_col.enhance(color)
    elif convert_flag==3:
        enh_bri=ImageEnhance.Brightness(image)
        image=enh_bri.enhance(brightness)
    elif convert_flag==4:
        image=image.transpose(Image.FLIP_LEFT_RIGHT)
        enh_col=ImageEnhance.Color(image)
        image=enh_col.enhance(color)
    elif convert_flag==5:
        image=image.transpose(Image.FLIP_LEFT_RIGHT)
        enh_bri=ImageEnhance.Brightness(image)
        image=enh_bri.enhance(brightness)
    elif convert_flag==6:
        enh_con=ImageEnhance.Contrast(image)
        image=enh_con.enhance(contrast)
    return image
def _main():
    for index in range(num):
        if os.listdir(os.path.join(path_trian_dir,str(index)))=='':
            continue
        if index<1:
            continue
        filename=os.listdir(os.path.join(path_trian_dir,str(index)))
        for image_filename in filename:
            filename_path=os.path.join(path_trian_dir+'/'+str(index),image_filename)
            image=Image.open(filename_path)
            print('Folder : %s processing: %s ' % (index,image_filename))
            for nums_index in range(nums_total):
                image_data=preprocess_image(image,nums_index)
                image_data.save(path_train_save_dir+'/'+str(index)+'/'+image_filename+'_'+str(nums_index+5)+'.jpg')
if __name__=='__main__':
    _main()