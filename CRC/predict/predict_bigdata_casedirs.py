#coding:utf-8
import os
import glob
import sys

sys.path.append("..")
from models import meancher_model
from utils_base import make_model_base
from predict_bigdata import predict_bigdata

if len(sys.argv)>=4:
    base_path=sys.argv[1]
    output_base_path=sys.argv[2]
    model_path=sys.argv[3]
else:
    base_path = '/media/disk2/pathology_data_process/group1/'
    output_base_path='/media/disk4/pathology_test0_save1/'
    model_path='../test0-save1.hdf5'

input_path=base_path+'common_images/'
output_path=output_base_path+'predict/'

dirs=glob.glob(input_path+'*')

if os.path.exists(output_path):
    print('Warning: the output_path has exist,continue proceed....')
 #   exit()
else:
    os.mkdir(output_path)

model = meancher_model()   #SSL
#model=make_model_base(True)   #SL
model.load_weights(model_path)

print('There are '+str(len(dirs))+' dirs')
for line in dirs:
    imags=glob.glob(line+'/*')
    if len(imags)==0:
        print('no image file at '+line)
        continue

    basename=os.path.basename(line)

    predictfile=output_path  + basename + '.txt'

    if os.path.exists(predictfile):
        print(predictfile+' has existed')
        continue

    print('processing ' + line + ' ********************')
    predict_bigdata(model, line , output_path +  basename + '.txt' )

    print('\n')


