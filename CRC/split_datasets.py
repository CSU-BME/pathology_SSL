import glob
import os
import numpy as np

input_path='/media/disk3/colon-semi-up3/datasets'       #path of input data
output_path='/media/disk3/colon-semi-up3/subs-datasets-0.1'
ratio=[0.1,0.6,0.3]    #trian,extra, test datasets
num_statistics=8
num_classes=2
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
        files = glob.glob(os.path.join(input_path, str(class_i)+'/*.jpg'))
        ids = [os.path.basename(file)[0:7] for file in files]
        ids = list(set(ids))
        np.random.shuffle(ids)

        num_test=int(len(ids)*ratio[2])         #test set with ration[2] ids.
        # get test sets
        output_path2 = os.path.join(output_path_t, 'test')
        os.makedirs(os.path.join(output_path2, str(class_i)))
        for id in ids[0:num_test]:  # test datasets
            command = 'cp ' + os.path.join(input_path, str(class_i) + '/') + \
                      id.strip() + '*.jpg ' + os.path.join(output_path2, str(class_i))
            os.system(command)

        output_path2 = os.path.join(output_path_t, 'train')
        if not os.path.exists(os.path.join(output_path2, str(class_i))):
            os.makedirs(os.path.join(output_path2, str(class_i)))

        output_path2 = os.path.join(output_path_t, 'extra')
        if not os.path.exists(os.path.join(output_path2, str(num_classes))):  # extra datasets with label=num_classes,i.e. no label
            os.makedirs(os.path.join(output_path2, str(num_classes)))

        # get train datasets with label and no label.
        train_ids=ids[num_test:]
        for id in train_ids:     #get one WSI
            basename=id+'*.jpg'
            files = glob.glob(os.path.join(input_path, str(class_i) + '/'+basename))
            np.random.shuffle(files)
            num_train = round(len(files) * ratio[0]/(1-ratio[2]))+1  # training set with label
            num_extra = len(files)-num_train  # extra sets
            num_val=0
            if val_ratio>0:
                num_val = round(num_train * val_ratio) # validation if need a validations set for hyperparameters

            output_path2 = os.path.join(output_path_t, 'train')   #train datasets
            for file in files[0:num_train]:
                command = 'cp "' + file +'" '+ os.path.join(output_path2, str(class_i))
                os.system(command)

            output_path2 = os.path.join(output_path_t, 'extra')
            for file in files[num_train:]:    #extra datasets
                command = 'cp "' +file +'" '+ os.path.join(output_path2, str(num_classes))
                os.system(command)

            if num_val>0:
                output_path2 = os.path.join(output_path_t, 'val')
                output_path_train=os.path.join(output_path_t, 'train')
                if not os.path.exists(os.path.join(output_path2, str(class_i))):
                    os.makedirs(os.path.join(output_path2, str(class_i)))

                for file in files[0:num_val]:     #get val from train datasets
                   file=os.path.basename(file)
                   command = 'mv ' + os.path.join(output_path_train, str(class_i) + '/"') + \
                          file+ '" ' + os.path.join(output_path2, str(class_i))
                   os.system(command)

for i in range(num_statistics):
    print('the %d extraction'%(i))
    create_random_datasets(i,ratio,val_ratio)


