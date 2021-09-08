from tensorflow.compat.v1.keras.applications import inception_v3
from tensorflow.compat.v1.keras import models
from tensorflow.compat.v1.keras import layers
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1.keras.callbacks import Callback
import tensorflow.compat.v1.keras.backend as KTF
from sklearn import metrics
import copy
from inception_preprocessing import preprocess_image

#model definition
un_loss=1
num_classes=2+1     #2 classs and 1 unknown class,no label
constant=1

def add_noise(input_data):
    batch_size=tf.shape(input_data)[0]

    def condition(data,i, n):
        return i < n

    def loop_body(data,i, n):
        temp=tf.keras.backend.expand_dims(preprocess_image(input_data[i], 299, 299,True),0)
        i+=1
        return tf.concat([data,temp],axis=0),i,n

    data = tf.keras.backend.expand_dims(preprocess_image(input_data[0], 299, 299, True), 0)
    i=tf.constant(1)
    n=batch_size
    data,i, n = tf.while_loop(condition, loop_body, [data,i, n],
                    shape_invariants=[tf.TensorShape([None,299,299,3]),tf.TensorShape([]),tf.TensorShape([])])
    return data

def preprocess_input(input_data):
    batch_size=tf.shape(input_data)[0]

    def condition(data,i, n):
        return i < n

    def loop_body(data,i, n):
        temp=tf.keras.backend.expand_dims(preprocess_image(input_data[i], 299, 299),0)
        i+=1
        return tf.concat([data,temp],axis=0),i,n

    data = tf.keras.backend.expand_dims(preprocess_image(input_data[0], 299, 299), 0)
    i=tf.constant(1)
    n=batch_size
    data,i, n = tf.while_loop(condition, loop_body, [data,i, n],
                    shape_invariants=[tf.TensorShape([None,299,299,3]),tf.TensorShape([]),tf.TensorShape([])])
    return data


def make_model_base(train_flag,init,name=None,is_student=False,training=False):
    if training:  # train and val of training mode
        input_data = tf.keras.Input(shape=(None, None, 3), name='image', dtype=tf.uint8)  # original images

        if is_student == True:  # if student and training mode, add noise
            input_data2 = layers.Lambda(lambda x: add_noise(x))(input_data)  # preprocessing image with noises
        else:
            input_data2 = layers.Lambda(lambda x: preprocess_input(x))(input_data)  # preprocessing image
    else:  # testing mode
        input_data = tf.keras.Input(shape=(None, None, 3), name='image')  # preprocessed image
        input_data2 = input_data

    conv_base = inception_v3.InceptionV3(weights=init,
                                         include_top=False, input_shape=(299, 299, 3))(input_data2)

    avepool1 = layers.GlobalAveragePooling2D()(conv_base)
    avepool1 = layers.Activation('relu')(avepool1)
    output = layers.Dense(num_classes-1, activation='softmax')(avepool1)
    model = models.Model(input_data, output, name=name)
    model.trainable = train_flag
    return model

def meancher_model(training=False):
    if training:
        input_data = tf.keras.Input(shape=(None, None, 3), name='image', dtype=tf.uint8)  # original images
    else:
        input_data = tf.keras.Input(shape=(None, None, 3), name='image')  # preprocessed image

    teacher = make_model_base(train_flag=False, name='teacher', init='imagenet', training=training)
    student = make_model_base(train_flag=True, name='student', init='imagenet', is_student=True, training=training)

    t_output = teacher(input_data)
    s_output = student(input_data)

    output = tf.keras.layers.Concatenate(axis=-1)([t_output, s_output])
    model = models.Model(input_data, output)

    return model

def get_ema_efficient(decay, step,constant=1):
    return min(1 - 1 / (step + constant), decay)

class ActivationLogger(Callback):
    def __init__(self, decay=0.99, has_move=True):
        self.decay = decay
        self.epoch = 0
        self.has_move = has_move

    def set_model(self, model):
        self.model = model

    def on_epoch_end(self, epoch, logs=None):
        weights_s = self.model.get_layer('student').get_weights()
        weights_t = self.model.get_layer('teacher').get_weights()
        self.epoch += 1

        if self.has_move:
            ema_coefficient = get_ema_efficient(self.decay, self.epoch, constant)

            for i in range(len(weights_s)):
                weights_t[i] = (1 - ema_coefficient) * weights_s[i] + ema_coefficient * weights_t[i]
        else:
            weights_t = copy.deepcopy(weights_s)

        self.model.get_layer('teacher').set_weights(weights_t)

def my_loss(y_true, y_pred):
    [y_pred_t, y_pred_s] = tf.split(y_pred, 2, axis=1)  # num_classes-1
    [y_true_t, y_true_s] = tf.split(y_true, 2, axis=1)  # num_classes
    [y_true_s_1, _] = tf.split(y_true_s, [num_classes - 1, 1], axis=1)  # num_classes-1

    cross_entroy = KTF.categorical_crossentropy(y_true_s_1, y_pred_s)  # student loss
    loss_1 = tf.where(tf.not_equal(tf.argmax(y_true_s, axis=-1), num_classes - 1), cross_entroy,
                      tf.zeros_like(cross_entroy))  # supervised loss of student loss

    loss_2 = tf.where(tf.equal(tf.argmax(y_true_s, axis=-1), num_classes - 1),
                      KTF.mean(KTF.square(y_pred_t - y_pred_s), axis=-1),  # student and teacher difference
                      tf.zeros_like(cross_entroy))  # unsupervised loss

    return loss_1 + loss_2 * un_loss

#accuracy of student model
def my_metrics(y_true, y_pred):
    [y_pred_t,y_pred_s]=tf.split(y_pred,2,axis=1)
    [y_true_t, y_true_s] = tf.split(y_true, 2,axis=1)

    preds=tf.argmax(y_pred_s,axis=-1)
    labels=tf.argmax(y_true_s,axis=-1)

    preds_0=preds[tf.equal(labels,0)]
    labels_0=labels[tf.equal(labels,0)]

    preds_1 = preds[tf.equal(labels, 1)]
    labels_1 = labels[tf.equal(labels, 1)]

    preds = tf.concat([preds_0, preds_1], axis=-1)
    labels = tf.concat([labels_0, labels_1], axis=-1)

    c1 = tf.reduce_sum(tf.cast(tf.equal(preds, labels), tf.float32))
    c2 = tf.cast(tf.shape(labels), tf.float32)

    return (c1+1e-10)/(c2+1e-10)

#accuracy of teacher model
def my_metrics_t(y_true, y_pred):
    [y_pred_t,y_pred_s]=tf.split(y_pred,2,axis=1)
    [y_true_t, y_true_s] = tf.split(y_true, 2,axis=1)

    preds=tf.argmax(y_pred_t,axis=-1)
    labels=tf.argmax(y_true_t,axis=-1)

    preds_0=preds[tf.equal(labels,0)]
    labels_0=labels[tf.equal(labels,0)]

    preds_1 = preds[tf.equal(labels, 1)]
    labels_1 = labels[tf.equal(labels, 1)]

    preds = tf.concat([preds_0, preds_1], axis=-1)
    labels = tf.concat([labels_0, labels_1], axis=-1)

    c1 = tf.reduce_sum(tf.cast(tf.equal(preds, labels), tf.float32))
    c2 = tf.cast(tf.shape(labels), tf.float32)

    return (c1+1e-10)/(c2+1e-10)

def my_metrics_same(y_true,y_pred):
    [y_pred_t, y_pred_s] = tf.split(y_pred, 2, axis=1)
    [y_true_t, y_true_s] = tf.split(y_true, 2, axis=1)

    preds_t = tf.argmax(y_pred_t, axis=-1)
    preds_s = tf.argmax(y_pred_s, axis=-1)

    c1 = tf.reduce_sum(tf.cast(tf.equal(preds_t, preds_s), tf.float32))
    c2 = tf.cast(tf.shape(preds_t), tf.float32)

    return (c1 + 1e-10) / (c2 + 1e-10)


def add_weight_decay(model, weight_decay):
    if (weight_decay is None) or (weight_decay == 0.0):
        return

    # recursion inside the model
    def add_decay_loss(m, factor):
        if isinstance(m, tf.keras.Model):
            for layer in m.layers:
                add_decay_loss(layer, factor)
        else:
            for param in m.trainable_weights:
                with tf.keras.backend.name_scope('weight_regularizer'):
                    regularizer =  tf.keras.regularizers.l2(factor)(param)
                    m.add_loss(lambda: regularizer)

    # weight decay and l2 regularization differs by a factor of 2
    add_decay_loss(model, weight_decay/2.0)
    return

def auc_t(y_true,y_pred):
    [y_true_t, y_true_s] = tf.split(y_true, 2, axis=1)

    y_pred = tf.cast(y_pred[:, 1], dtype=tf.float32)
    y_true = tf.cast(tf.argmax(y_true_t[:,:2], axis=-1), dtype=tf.float32)
    y_true_f = KTF.flatten(y_true)
    y_pred_f = KTF.flatten(y_pred)
    return tf.numpy_function(metrics.roc_auc_score, (y_true_f, y_pred_f), tf.double)

def auc_s(y_true,y_pred):
    [y_true_t, y_true_s] = tf.split(y_true, 2, axis=1)

    y_pred = tf.cast(y_pred[:, 3], dtype=tf.float32)
    y_true = tf.cast(tf.argmax(y_true_s[:,:2], axis=-1), dtype=tf.float32)
    y_true_f = KTF.flatten(y_true)
    y_pred_f = KTF.flatten(y_pred)

    return tf.numpy_function(metrics.roc_auc_score, (y_true_f, y_pred_f), tf.double)

