from tensorflow.compat.v1.keras import optimizers
from utils import make_test_dataset,meancher_model,make_model_base
import numpy as np
from sklearn import metrics
import os
import glob

def predict_MT_AUC(model,model_file,test_path_pos,test_path_neg,batch_size):
    file_path = model_file

    if os.path.exists(file_path):
        model.load_weights(file_path)
        print("checkpoint_loaded")

    labels = []
    preds = []
    test_dataset = make_test_dataset(test_path_neg,batch_size)

    preds_neg = model.predict(test_dataset)
    preds_neg = list(preds_neg[:, 4])   #4 for student model, 2 for teacher model
    labels_neg = list(np.zeros(len(preds_neg)))
    labels.extend(labels_neg)
    preds.extend(preds_neg)

    test_dataset = make_test_dataset(test_path_pos, batch_size)

    preds_pos = model.predict(test_dataset)
    preds_pos = list(preds_pos[:, 4])
    labels_pos = list(np.ones(len(preds_pos)))

    labels.extend(labels_pos)
    preds.extend(preds_pos)

    auc = metrics.roc_auc_score(labels, preds)
    return auc

def predict_AUC(model,model_file,test_path_pos,test_path_neg,batch_size):
    file_path = model_file

    if os.path.exists(file_path):
        model.load_weights(file_path)
        print("checkpoint_loaded")

    labels = []
    preds = []
    test_dataset = make_test_dataset(test_path_neg,batch_size)

    preds_neg = model.predict(test_dataset)
    preds_neg = list(np.argmax(preds_neg, axis=-1))
    labels_neg = list(np.zeros(len(preds_neg)))
    labels.extend(labels_neg)
    preds.extend(preds_neg)

    test_dataset = make_test_dataset(test_path_pos,batch_size)
    preds_pos = model.predict(test_dataset)
    preds_pos = list(np.argmax(preds_pos, axis=-1))
    labels_pos = list(np.ones(len(preds_pos)))

    labels.extend(labels_pos)
    preds.extend(preds_pos)
    auc = metrics.roc_auc_score(labels, preds)
    return auc



#patch-level evaluate
#Dataset-PATT(testing set)
batch_size=128
test_path_neg='data/test-0-tfrecord.tfr'     #neg
test_path_pos='data/test-1-tfrecord.tfr'     #pos
print('evlaute models on Dataset-PAT')

#evaluate SSL at patch_level
model_files=glob.glob('pre-models/Model-5%-SSL.hdf5')
aucs=[]
for model_file in model_files:
    print('evaluate the model %s'%(os.path.basename(model_file)))
    model = meancher_model()
    auc=predict_MT_AUC(model,model_file,test_path_pos,test_path_neg,batch_size)
    auc=(os.path.basename(model_file),auc)
    aucs.append(auc)
    print(auc)

model_files=glob.glob('pre-models/Model-10%-SSL.hdf5')
for model_file in model_files:
    print('evaluate the model %s'%(os.path.basename(model_file)))
    model = meancher_model()
    auc = predict_MT_AUC(model, model_file, test_path_pos, test_path_neg, batch_size)
    auc = (os.path.basename(model_file), auc)
    aucs.append(auc)
    print(auc)

#evaluate SL models
model_files=glob.glob('pre-models/Model-5%-SL.hdf5')
for model_file in model_files:
    print('evaluate the model %s'%(os.path.basename(model_file)))
    model = make_model_base(True)
    auc = predict_AUC(model, model_file, test_path_pos, test_path_neg, batch_size)
    auc = (os.path.basename(model_file), auc)
    print(auc)
    aucs.append(auc)

#evaluate SL models
model_files=glob.glob('pre-models/Model-10%-SL.hdf5')
for model_file in model_files:
    print('evaluate the model %s'%(os.path.basename(model_file)))
    model = make_model_base(True)
    auc = predict_AUC(model, model_file, test_path_pos, test_path_neg, batch_size)
    auc = (os.path.basename(model_file), auc)
    aucs.append(auc)
    print(auc)

print('************************\n')
print(aucs)
print('************************\n')

#Dataset-PAT
test_path_neg='data/NCT-CRC-HE-100K-NONORM2-0-tfrecord.tfr'   #neg
test_path_pos='data/NCT-CRC-HE-100K-NONORM2-1-tfrecord.tfr'    #pos
batch_size=128
print('evlaute models on Dataset-PAT')
#evaluate SSL at patch_level
aucs=[]
model_files=glob.glob('pre-models/Model-5%-SSL.hdf5')
for model_file in model_files:
    print('evaluate the model %s'%(os.path.basename(model_file)))
    model = meancher_model()
    auc=predict_MT_AUC(model,model_file,test_path_pos,test_path_neg,batch_size)
    auc = (os.path.basename(model_file), auc)
    aucs.append(auc)
    print(auc)

model_files=glob.glob('pre-models/Model-10%-SSL.hdf5')
for model_file in model_files:
    print('evaluate the model %s'%(os.path.basename(model_file)))
    model = meancher_model()
    auc = predict_MT_AUC(model, model_file, test_path_pos, test_path_neg, batch_size)
    auc = (os.path.basename(model_file), auc)
    aucs.append(auc)
    print(auc)

#evaluate SL models
model_files=glob.glob('pre-models/Model-5%-SL.hdf5')
for model_file in model_files:
    print('evaluate the model %s'%(os.path.basename(model_file)))
    model = make_model_base(True)
    auc = predict_AUC(model, model_file, test_path_pos, test_path_neg, batch_size)
    auc = (os.path.basename(model_file), auc)
    print(auc)
    aucs.append(auc)

#evaluate SL models
model_files=glob.glob('pre-models/Model-10%-SL.hdf5')
for model_file in model_files:
    print('evaluate the model %s'%(os.path.basename(model_file)))
    model = make_model_base(True)
    auc = predict_AUC(model, model_file, test_path_pos, test_path_neg, batch_size)
    auc = (os.path.basename(model_file), auc)
    aucs.append(auc)
    print(auc)

print('************************\n')
print(aucs)
print('************************\n')


