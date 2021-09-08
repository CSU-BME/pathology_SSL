from models import add_weight_decay
from tensorflow.compat.v1.keras import optimizers
from utils_base import make_datasets,make_model_base,auc
from tensorflow.compat.v1.keras.callbacks import ModelCheckpoint,EarlyStopping, LearningRateScheduler
import os

def train_models_subs(train_path,val_path,test_path,batch_size,file_path):
    train_dataset, val_dataset, test_dataset = make_datasets(train_path, val_path, test_path, batch_size)
    model = make_model_base(True)
    model.summary()

    add_weight_decay(model, 0.0001)
    def exp_decay(epoch):
        # lrate = learning_rate * np.exp(-decay_rate*epoch)
        lrate = learning_rate * pow(decay_rate, epoch)
        return lrate

    callbacklist = [ModelCheckpoint(file_path, monitor='val_accuracy',
                                    verbose=1, save_best_only=True, save_weights_only=True, mode='max'),
                    EarlyStopping(monitor='val_accuracy', patience=50),
                    LearningRateScheduler(exp_decay),
                    ]

    model.compile(optimizer=optimizers.Adam(lr=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_dataset, epochs=epochs, steps_per_epoch=steps_per_epoch, callbacks=callbacklist,
              validation_data=val_dataset, validation_steps=validation_steps, verbose=1)

    # for val
    if os.path.exists(file_path):
        model.load_weights(file_path)
        print("checkpoint_reloaded")
    else:
        print('error! not found model: '+file_path)
        return

    model.compile(optimizer=optimizers.Adam(lr=learning_rate), loss='categorical_crossentropy', metrics=[auc])
    eval_results = model.evaluate(test_dataset)
    return (eval_results[1])

batch_size = 64
epochs = 500
steps_per_epoch = 100
decay_rate = 0.99
learning_rate = 1e-3
validation_steps = 10
path='/media/disk3/colon-semi-up3/subs-datasets-0.1' #subs-datasets-0.1,0.05
model_path='Model-0.1-SL'
fw=open('subs-colon-0.1-SL.txt','a')

for index_j in range(8):
    print('process dataset:'+str(index_j))
    train_path = os.path.join(path, 'train-sub'+str(index_j)+'.tfr')  # data path
    val_path = os.path.join(path, 'val-sub'+str(index_j)+'.tfr')
    test_path = os.path.join(path, 'test-sub'+str(index_j)+'.tfr')
    file_path = model_path+'-'+str(index_j)+'.hdf5'
    acc=train_models_subs(train_path,val_path,test_path,batch_size,file_path)
    fw.write('the %d dataset AUC is: %s\n'%(index_j,acc))
    fw.flush()

fw.close()
