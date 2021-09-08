import glob
import os
import numpy as np

input_path='/media/disk3/lung-semi-up/lung-3class'       #path of input data
output_path='/media/disk3/lung-semi-up/subs-lung-3class-0.2'
ratio=[0.2,0.6,0.2]    #     #ratio of trian,extra, test datasets
num_statistics=8
num_classes=3
val_ratio=0.1   #get some data from training set for validation

def create_random_datasets(index,ratio,val_ratio=0):       #val_ratio:  get val data from training set for validations
    output_path_t=os.path.join(output_path,str(index))
    if os.path.exists(output_path_t):
        print("need a new path")
        exit()

    os.makedirs(output_path_t)
    os.makedirs(os.path.join(output_path_t, 'train'))
    os.makedirs(os.path.join(output_path_t, 'extra'))
    os.makedirs(os.path.join(output_path_t, 'test'))
    os.makedirs(os.path.join(output_path_t, 'val'))
    for class_i in range(num_classes):
        files = glob.glob(os.path.join(input_path, str(class_i)+'/*.jpeg'))
        np.random.shuffle(files)

        num_train = int(len(files) * ratio[0])    #training set
        num_extra=int(len(files) * ratio[1])      #extra sets
        num_val=int(len(files)*ratio[0]*val_ratio)     #validation if need a validations set for hyperparameters

        output_path2 = os.path.join(output_path_t, 'train')
        os.makedirs(os.path.join(output_path2, str(class_i)))
        for file in files[0:num_train]:
            id = file.split('/')[-1]
            command = 'cp ' + os.path.join(input_path, str(class_i)+'/') + \
                  id + ' ' + os.path.join(output_path2, str(class_i))
            os.system(command)

        output_path2 = os.path.join(output_path_t, 'extra')
        if not os.path.exists(os.path.join(output_path2, str(num_classes))):   #extra datasets with label=num_classes,i.e. no label
            os.makedirs(os.path.join(output_path2, str(num_classes)))
        for file in files[num_train:num_train+num_extra]:    #extra datasets
            id = file.split('/')[-1]
            command = 'cp ' + os.path.join(input_path, str(class_i) + '/') + \
                      id + ' ' + os.path.join(output_path2, str(num_classes))
            os.system(command)

        output_path2 = os.path.join(output_path_t, 'test')
        os.makedirs(os.path.join(output_path2, str(class_i)))
        for file in files[num_train + num_extra:]:       #test datasets
            id=file.split('/')[-1]
            command = 'cp ' + os.path.join(input_path, str(class_i) + '/') + \
                      id + ' ' + os.path.join(output_path2, str(class_i))
            os.system(command)

        if num_val>0:
            output_path2 = os.path.join(output_path_t, 'val')
            output_path_train=os.path.join(output_path_t, 'train')
            os.makedirs(os.path.join(output_path2, str(class_i)))

            for file in files[0:num_val]:
                id = file.split('/')[-1]
                command = 'mv ' + os.path.join(output_path_train, str(class_i) + '/') + \
                          id + ' ' + os.path.join(output_path2, str(class_i))
                os.system(command)


for i in range(num_statistics):
    print('the %d extraction'%(i))
    create_random_datasets(i,ratio,val_ratio)


