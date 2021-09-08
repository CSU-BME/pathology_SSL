from tensorflow.compat.v1.keras import optimizers
from utils_base import make_test_dataset,make_model_base,log_metrics
import os
import glob

path='/media/disk3/lung-semi-up/subs-lung-3class-0.2'    #data path
model_path='../test/models-new/lung/Model-0.2-SL'      #model base name
fw=open('subs-lung-0.2-SL-evaluate-new.txt','a')    #result file

model_list=glob.glob(model_path+'*.hdf5')
model_list.sort()
batch_size=128

accuracy=[]

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

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'])
    print('evaluate the model: ' + filepath)
    eval_results=model.evaluate(test_dataset)
    del test_dataset

    fw.write('the %d dataset accuracy is: %s\n'%(index_j,eval_results[1]))
    fw.flush()
    accuracy.extend([eval_results[1]])
    print('\n')
    print('evaluate %s is finished. '%(filepath))
    print('\n')

fw.write('the accuracy:'+str(accuracy))
fw.write('\n')
fw.close()







