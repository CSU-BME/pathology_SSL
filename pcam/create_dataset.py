import glob
import os
import numpy as np
from keras.utils import HDF5Matrix
import cv2

def get_data(path):
    x_train = HDF5Matrix(os.path.join(path,'camelyonpatch_level_2_split_train_x.h5'), 'x')[0:]
    y_train = HDF5Matrix(os.path.join(path,'camelyonpatch_level_2_split_train_y.h5'),'y')[0:]
    y_train=np.squeeze(y_train)
    x_valid = HDF5Matrix(os.path.join(path,'camelyonpatch_level_2_split_valid_x.h5'), 'x')[0:]
    y_valid = HDF5Matrix(os.path.join(path,'camelyonpatch_level_2_split_valid_y.h5'), 'y')[0:]
    y_valid=np.squeeze(y_valid)
    x_test = HDF5Matrix(os.path.join(path,'camelyonpatch_level_2_split_test_x.h5'),'x')[0:]
    y_test = HDF5Matrix(os.path.join(path,'camelyonpatch_level_2_split_test_y.h5'),'y')[0:]
    y_test=np.squeeze(y_test)

    return x_train,y_train,x_valid,y_valid,x_test,y_test

def create_one_dataset(x,y,key,output_path):
    os.makedirs(os.path.join(output_path,key))
    classes=max(y)+1

    for i in range(classes):
        os.makedirs(os.path.join(os.path.join(output_path,key),str(i)))

    for i in range(len(x)):
        img=x[i]
        img=img[:,:,(2,1,0)]
        label=y[i]

        path=os.path.join(os.path.join(output_path,key),str(label))
        pathname=os.path.join(path,str(i)+'.jpg')
        cv2.imwrite(pathname,img)

        if i%500==0:
            print('process the %d image'%(i))

input_labels='/media/disk3/lung-pcam/PCam'
input_files='/media/disk3/lung-pcam/PCam'
output_path='/media/disk3/pcam-semi-up/PCam_output'

x_train,y_train,x_valid,y_valid,x_test,y_test=get_data(input_labels)   #get images and labels
create_one_dataset(x_train,y_train,'train',output_path)    #create datasets of images and labels
create_one_dataset(x_valid,y_valid,'valid',output_path)
create_one_dataset(x_test,y_test,'test',output_path)









