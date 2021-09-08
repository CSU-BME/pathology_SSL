from models import meancher_model,my_loss,ActivationLogger,my_metrics,add_weight_decay,\
                                my_metrics_t,my_metrics_same,auc_t,auc_s
from tensorflow.compat.v1.keras import optimizers
from utils_v2 import make_datasets
from tensorflow.compat.v1.keras.callbacks import ModelCheckpoint,EarlyStopping
import os

def train_models_subs(train_path,extra_path,val_path,test_path,batch_size,file_path):
    train_dataset, val_dataset, test_dataset = make_datasets(train_path,
                                                             val_path, test_path, batch_size, extra_path)
    model = meancher_model(training=True)
    model.summary()

    if os.path.exists(file_path):
        model.load_weights(file_path)
        print("checkpoint_loaded")

    add_weight_decay(model, 0.0001)
    callback_1 = ActivationLogger(0.95,True)   #0.9, 0.95,0.99

    def exp_decay(epoch):
        lrate = learning_rate * pow(decay_rate, epoch)
        return lrate

    callbacklist = [ModelCheckpoint(file_path, monitor='val_my_metrics',
                                    verbose=1, save_best_only=True, save_weights_only=True, mode='max'),
                    callback_1,
                    EarlyStopping(monitor='val_my_metrics', patience=50),  #50 or 30 for faster training
                    #LearningRateScheduler(exp_decay),
                    ]

    model.compile(optimizer=optimizers.Adam(lr=learning_rate), loss=my_loss, metrics=[my_metrics,my_metrics_t,my_metrics_same])
    model.fit(train_dataset, epochs=first_epoches, steps_per_epoch=steps_per_epoch, callbacks=[callback_1],
              validation_data=val_dataset, validation_steps=validation_steps, verbose=1)

    model.fit(train_dataset, epochs=epochs - first_epoches, steps_per_epoch=steps_per_epoch,callbacks=callbacklist,
              validation_data=val_dataset, validation_steps=validation_steps, verbose=1)

    # for val
    model = meancher_model()
    if os.path.exists(file_path):
        model.load_weights(file_path)
        print("checkpoint_reloaded")
    else:
        print('error! not found model: '+file_path)
        return

    model.compile(optimizer=optimizers.Adam(lr=learning_rate), loss=my_loss, metrics=[auc_t,auc_s])
    eval_results = model.evaluate(test_dataset)
    return(eval_results[1],eval_results[2])

batch_size = 128
epochs = 500
steps_per_epoch = 100
first_epoches = 50 #pretrain and don't save model
decay_rate = 0.99
learning_rate = 1e-4
validation_steps = 10
path='/media/disk3/share/subs-datasets-0.1' #subs-datasets-0.1,0.05
model_path='Model-0.1-SSL'
fw=open('subs-colon-0.1-SSL.txt','a')

for index_j in range(8):
    train_path = os.path.join(path, 'train-sub'+str(index_j)+'.tfr')  # data path
    val_path = os.path.join(path, 'val-sub'+str(index_j)+'.tfr')
    extra_path = os.path.join(path, 'extra-sub'+str(index_j)+'.tfr')
    test_path = os.path.join(path, 'test-sub'+str(index_j)+'.tfr')
    file_path = model_path+'-'+str(index_j)+'.hdf5'
    acc1,acc2=train_models_subs(train_path,extra_path,val_path,test_path,batch_size,file_path)
    fw.write('the %d dataset AUC is: T:%s,  S:%s\n'%(index_j,acc1,acc2))
    fw.flush()

fw.close()





