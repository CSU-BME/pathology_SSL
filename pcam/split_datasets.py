import glob
import os
import numpy as np

#split training set by ratio.
input_path='/media/disk3/pcam-semi-up/PCam_output/'       #path of training set of pcam.
output_path='/media/disk3/pcam-semi-up/subs-pcam-0.01'
ratio=[0.01,0.99]    #ratio of labeled trainning set, unlabeled traininig set from training set.
                   # ratio[0] is also the extration ratio of validation set
num_statistics=8
num_classes=2

def create_random_datasets(index,ratio):       #val_ratio:  get val data from training set for validations
    output_path_t=os.path.join(output_path,str(index))
    if os.path.exists(output_path_t):
        print("need a new path")
        exit()

    os.makedirs(output_path_t)
    os.makedirs(os.path.join(output_path_t, 'train'))
    os.makedirs(os.path.join(output_path_t, 'extra'))

    for class_i in range(num_classes):
        input_path2=os.path.join(input_path,'train')      #training set  path
        files = glob.glob(os.path.join(input_path2, str(class_i)+'/*.jpg'))
        np.random.shuffle(files)

        num_train = int(len(files) * ratio[0])    #training set
        num_extra=int(len(files) * ratio[1])      #extra traning sets with no label

        output_path2 = os.path.join(output_path_t, 'train')
        os.makedirs(os.path.join(output_path2, str(class_i)))
        for file in files[0:num_train]:
            command = 'cp ' + file +' ' + os.path.join(output_path2, str(class_i))
            os.system(command)

        output_path2 = os.path.join(output_path_t, 'extra')
        if not os.path.exists(os.path.join(output_path2, str(num_classes))):   #extra datasets with label=num_classes,i.e. no label
            os.makedirs(os.path.join(output_path2, str(num_classes)))
        for file in files[num_train:num_train+num_extra]:    #extra datasets
            command = 'cp ' + file +' ' + os.path.join(output_path2, str(num_classes))
            os.system(command)

        #copy validation and test for every datasets.
        output_path2 = os.path.join(output_path_t, 'test')
        input_path2=os.path.join(input_path,'test')
        os.makedirs(os.path.join(output_path2, str(class_i)))
        files = glob.glob(os.path.join(input_path2, str(class_i) + '/*.jpg'))

        for file in files:
            command = 'cp ' + file + ' ' + os.path.join(output_path2, str(class_i))
            os.system(command)

        output_path2 = os.path.join(output_path_t, 'valid')
        input_path2 = os.path.join(input_path, 'valid')
        os.makedirs(os.path.join(output_path2, str(class_i)))
        files = glob.glob(os.path.join(input_path2, str(class_i) + '/*.jpg'))
        np.random.shuffle(files)
        num_valid = int(len(files) * ratio[0])  # the extration validation set by ratio

        for file in files[0:num_valid]:
            command = 'cp ' + file + ' ' + os.path.join(output_path2, str(class_i))
            os.system(command)

for i in range(num_statistics):
    print('the %d extraction'%(i))
    create_random_datasets(i,ratio)


