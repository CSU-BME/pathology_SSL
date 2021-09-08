#coding:utf-8
import tensorflow as tf
from os import listdir
from os.path import isfile, join
import os
import pickle
import cv2
import numpy as np
import math


def get_test_images(mypath):
     return [mypath + '/' + f for f in listdir(mypath) if isfile(join(mypath, f)) and (f.find('.jpg') != -1  or f.find('.jpeg') != -1 )]

def transform_img_fn(image_file,patch_mask,beg_height,beg_width,heinum,widnum, \
        sampleheight,offsethei,samplewidth,offsetwid):
    images = []
    patch_name=[]

    #read the image data
  #  image_raw = tf.image.decode_jpeg(open(image_file, 'rb').read(), channels=3)
    image_raw=cv2.imread(image_file)
    if image_raw is None:
        return None,[]
    image_raw = image_raw[:, :, (2, 1, 0)]

    for ii in range(heinum):
        for jj in range(widnum):
            if patch_mask[beg_height+ii,beg_width+jj]==False:
                continue

            #get center region
            patch_raw = image_raw[ii * sampleheight+offsethei:(ii + 1) * sampleheight-offsethei, jj * samplewidth+offsetwid:(jj + 1) * samplewidth-offsetwid, :]
            patch_raw = ((patch_raw / 255.0)-0.5)*2
            #patch_raw=((patch_raw/255.0)-train_mean)*1.4
            #patch_raw=np.clip(patch_raw,-1,1)

            #patch_raw=patch_raw[0:299,0:299,:]
            patch_raw = cv2.resize(patch_raw, (299, 299))
            #save image data
            images.append(patch_raw)
            patch_name.append([str(beg_height + ii), str(beg_width + jj)])

    return np.array(images),patch_name


def predict_bigdata(model,test_path,output):
    # get image info
    basename = os.path.basename(test_path)
    fto = open(output, 'w')

    test_batch = 2000
    print('the batch patches size  is ' + str(test_batch))

    fw_info = open(os.path.join(test_path, basename + '_info'), 'rb')
    split_height = pickle.load(fw_info)  # image count at height
    split_width = pickle.load(fw_info)  # image count at width
    sampleheight = pickle.load(fw_info)  # patch height
    samplewidth = pickle.load(fw_info)  # patch width
    patch_mask = pickle.load(fw_info)  # save mask of patches and background
    saved_dict = pickle.load(fw_info,encoding='utf-8')

    central_fraction = 1   #crop center data
    offsetwid=int(samplewidth * (1 -central_fraction)/2)
    offsethei=int(sampleheight*(1-central_fraction)/2)

    inimage_height=sampleheight-2*offsethei
    inimage_width=samplewidth-2*offsetwid

    print('the images files in the case are: '+str(split_width*split_height))

    count=0
    beg_width=0    #beginning patch index at width
    beg_height=0
    heinum=0
    widnum=0
    tissue_patch=None
    patch_list=[]
    for ii in range(split_height):    #process all the image files
        for jj in range(split_width):
            print('begin read and transform the  %d image file'%(count+1))
            saved_name = basename + '_' + str(ii) + '_' + str(jj) + '.jpg'

            [heinum,widnum]=saved_dict[saved_name].split('_')
            heinum=int(heinum)//sampleheight   #patches num at height
            widnum=int(widnum)//samplewidth

            #read image data and tissue patches list
            temp_patch,temp_list=transform_img_fn(os.path.join(test_path,saved_name),patch_mask,beg_height,beg_width,heinum,widnum,
                                                  sampleheight,offsethei,samplewidth,offsetwid)

            if temp_patch is not None and len(temp_patch.shape) == 4:
               if tissue_patch is None:
                   tissue_patch=temp_patch
               else:
                   tissue_patch=np.concatenate((tissue_patch,temp_patch),axis=0)

               patch_list.extend(temp_list)

            count += 1  # image count
            beg_width = beg_width + widnum

        beg_height = beg_height + heinum
        beg_width=0

    if tissue_patch is None:
        test_num = 0
    else:
        test_num = math.ceil(len(patch_list) / test_batch)

    for batch_i in range(test_num):
        #process patches in one image file, one batch one times
        print('Start doing predictions: the ' + str(batch_i + 1) + '  batch')
        preds = model.predict(tissue_patch[batch_i*test_batch:(batch_i+1)*test_batch])   #get possibility of patches in the batch
        preds=preds[:,0:2]
        for p in range(len(preds)):
            image_info=basename+'_'+patch_list[p + batch_i * test_batch][0]+'_'+patch_list[p + batch_i * test_batch][1]
            fto.write(image_info)    #write patch indexes

            for j in range(len(preds[p, :])):
                fto.write('\t' + str(preds[p, j]))

            fto.write('\n')

    fto.close()
    fw_info.close()
