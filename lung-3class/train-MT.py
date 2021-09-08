from models import meancher_model,my_loss,ActivationLogger,my_metrics,add_weight_decay,\
                                    my_metrics_t,my_metrics_same
from tensorflow.compat.v1.keras import optimizers
from utils_v2 import make_datasets,log_metrics
from tensorflow.compat.v1.keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
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
    callback_1 = ActivationLogger(0.9,True)

    def exp_decay(epoch):
        lrate = learning_rate * pow(decay_rate, epoch)
        return lrate

    callbacklist = [ModelCheckpoint(file_path, monitor='val_my_metrics',
                                    verbose=1, save_best_only=True, save_weights_only=True, mode='max'),
                    EarlyStopping(monitor='val_my_metrics', patience=100),
                    callback_1
                    #LearningRateScheduler(exp_decay),
                    ]

    model.compile(optimizer=optimizers.Adam(lr=learning_rate), loss=my_loss, metrics=[my_metrics,my_metrics_t,my_metrics_same])
    model.fit(train_dataset, epochs=first_epoches, steps_per_epoch=steps_per_epoch, callbacks=[callback_1],
              validation_data=val_dataset, validation_steps=validation_steps, verbose=1)

    model.fit(train_dataset, epochs=epochs - first_epoches, steps_per_epoch=steps_per_epoch, callbacks=callbacklist,
              validation_data=val_dataset, validation_steps=validation_steps, verbose=1)

    model = meancher_model()
    if os.path.exists(file_path):
        model.load_weights(file_path)
        print("checkpoint_reloaded")
    else:
        print('error! not found model: '+file_path)
        return

    model.compile(optimizer=optimizers.Adam(lr=learning_rate), loss=my_loss, metrics=[my_metrics_t,my_metrics])
    eval_results = model.evaluate(test_dataset)
    return (eval_results[1], eval_results[2])

batch_size = 32  #32,64,128
epochs = 500
steps_per_epoch = 100
first_epoches = 150
decay_rate = 0.99
learning_rate = 1e-4
validation_steps = 8  #2，4，8  for 0.05，0.1，0.2
path='/media/disk3/lung-semi-up/subs-lung-3class-0.2' #subs-lung-3class-0.05，subs-lung-3class-0.8-no-extra
model_path='Model-0.2-SSL'
fw=open('subs-lung-3class-0.2-SSL.txt','a')

for index_j in range(8):
    train_path = os.path.join(path, 'train-sub'+str(index_j)+'.tfr')  # data path
    val_path = os.path.join(path, 'val-sub'+str(index_j)+'.tfr')
    extra_path = os.path.join(path, 'extra-sub'+str(index_j)+'.tfr')
    test_path = os.path.join(path, 'test-sub'+str(index_j)+'.tfr')
    file_path = model_path+'-'+str(index_j)+'.hdf5'
    acc1,acc2=train_models_subs(train_path,extra_path,val_path,test_path,batch_size,file_path)
    fw.write('the %d dataset accuracy is: T:%s,  S:%s\n'%(index_j,acc1,acc2))
    fw.flush()

fw.close()





