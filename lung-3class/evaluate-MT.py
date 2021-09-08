from models import meancher_model,my_loss,ActivationLogger,my_metrics,my_metrics_t
from tensorflow.compat.v1.keras import optimizers
from utils_v2 import make_test_dataset
import os
import glob
import numpy as np

path='/media/disk3/lung-semi-up/subs-lung-3class-0.2'
model_path='../test/models-new/lung/Model-0.2-SSL'
fw=open('subs-lung-0.2-SSL-evaluate-new.txt','a')

model_list=glob.glob(model_path+'*.hdf5')
model_list.sort()
batch_size=128

accuracy=[]   #[n,2] for accuracy of  teacher and student
for filepath in model_list:
    index_j=int(filepath.split('-')[-1].split('.')[0])
    test_path = os.path.join(path, 'test-sub'+str(index_j)+'.tfr')
    test_dataset = make_test_dataset(test_path, batch_size)

    model = meancher_model()

    if os.path.exists(filepath):
        model.load_weights(filepath)
        print("the model %s loaded"%(filepath))
    else:
        print('error! not found model: '+filepath)
        fw.write('error! not found model: '+filepath+'\n')
        fw.flush()
        continue

    model.compile(optimizer=optimizers.Adam(lr=1e-4), loss=my_loss, metrics=[my_metrics_t,my_metrics])
    print('evaluate the model: '+filepath)
    eval_results = model.evaluate(test_dataset)
    del test_dataset

    fw.write('the %d dataset accuracy is: T:%s,  S:%s \n'  \
             %(index_j,eval_results[1],eval_results[2]))
    fw.flush()
    accuracy.append([eval_results[1],eval_results[2]])
    print('\n')
    print('evaluate %s is finished. '%(filepath))
    print('\n')

accuracy=np.array(accuracy)
fw.write('the accuracy T:'+str(accuracy[:,0]))
fw.write('\n')
fw.write('the accuracy S:'+str(accuracy[:,1]))
fw.write('\n')

fw.close()




