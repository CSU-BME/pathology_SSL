from tensorflow.compat.v1.keras import optimizers
from utils_base import make_test_dataset,make_model_base,auc
import os
import glob

path='/media/disk3/colon-semi-up3/subs-datasets-0.1'    #data path
model_path='Model-0.1-SL'      #model base name
fw=open('subs-colon-0.1-SL-evaluate.txt','a')    #result file

model_list=glob.glob(model_path+'*.hdf5')
model_list.sort()
batch_size=64

accuracy=[]
AUC=[]

for filepath in model_list:
    index_j=int(filepath.split('-')[-1].split('.')[0])
    test_path = os.path.join(path, 'test-sub'+str(index_j)+'.tfr')
    test_dataset = make_test_dataset(test_path, batch_size)

    model = make_model_base(True)

    if os.path.exists(filepath):
        model.load_weights(filepath)
        print("the model %s loaded"%(filepath))
    else:
        print('error! not found model: '+filepath)
        fw.write('error! not found model: '+filepath+'\n')
        fw.flush()
        continue

    model.compile(loss='categorical_crossentropy', metrics=[auc])
    print('evaluate the model: ' + filepath)
    eval_results=model.evaluate(test_dataset)
    del test_dataset

    fw.write('the %d dataset AUC is: %s\n'%(index_j,eval_results[1]))
    fw.flush()
    AUC.extend([eval_results[1]])
    print('\n')
    print('evaluate %s is finished. '%(filepath))
    print('\n')

fw.write('the AUC: '+str(AUC))
fw.write('\n')
fw.close()



