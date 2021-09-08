from models import meancher_model,my_loss,ActivationLogger,my_metrics,my_metrics_t,auc_s,auc_t
from tensorflow.compat.v1.keras import optimizers
from utils_v2 import make_test_dataset,log_metrics
import os
import glob
import numpy as np

path='/media/disk3/colon-semi-up3/subs-datasets-0.1'
model_path='Model-0.1-SSL'
fw=open('subs-colon-0.1-SSL-evaluate.txt','a')

model_list=glob.glob(model_path+'*.hdf5')
model_list.sort()
batch_size=128

AUC=[]   #[n,2] for AUC of  teacher and student
for filepath in model_list:
    index_j=int(filepath.split('-')[-1].split('.')[0])
    test_path = os.path.join(path, 'test-sub'+str(index_j)+'.tfr')
    #test_path = '/media/disk3/colon-semi-up/test/NCT-CRC-HE-100K-NONORM2-rand-tfrecord.tfr'    #NCT test dataset
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

    model.compile(optimizer=optimizers.Adam(lr=1e-4), loss=my_loss, metrics=[auc_t,auc_s])
    print('evaluate the model: '+filepath)
    eval_results = model.evaluate(test_dataset)
    del test_dataset

    fw.write('the %d dataset AUC is: T:%s, S:%s\n'  \
             %(index_j,eval_results[1],eval_results[2]))
    fw.flush()
    AUC.append([eval_results[1],eval_results[2]])
    print('\n')
    print('evaluate %s is finished. '%(filepath))
    print('\n')

AUC=np.array(AUC)
fw.write('the AUC T:'+str(AUC[:,0]))
fw.write('the AUC S:'+str(AUC[:,1]))
fw.write('\n')

fw.close()




