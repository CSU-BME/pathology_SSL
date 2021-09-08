import tensorflow.compat.v1 as tf
from inception_preprocessing import preprocess_image
from tensorflow.compat.v1.keras import models
from tensorflow.compat.v1.keras.applications import inception_v3
from tensorflow.compat.v1.keras import layers

def load_dataset(filename):
  return tf.data.TFRecordDataset([filename])

def parse_example_val(example_proto):
    #Extracts relevant fields from the `example_proto`.
    feature_spec = {
        'id': tf.FixedLenFeature((), tf.string, default_value=''),
        'image':
            tf.FixedLenFeature((), tf.string, default_value=''),
        'label':
            tf.FixedLenFeature((), tf.int64, default_value=-1),
    }

    features = tf.io.parse_single_example(example_proto, feature_spec)

    for key, item in features.items():
        if key=='image':
            features[key] = tf.image.decode_jpeg(features[key])
            features[key]=preprocess_image(features[key], 299, 299)    #no augmentation

    features.pop('id')
    features['label'] = tf.one_hot(features['label'], 3)
    features['label']=tf.concat([features['label'],features['label']],axis=0)   #add false label
    labels = features.pop('label')

    return features, labels


def make_test_dataset(file_path, batch_size):
    dataset = load_dataset(file_path)
    dataset = dataset.map(parse_example_val)
    dataset=dataset.shuffle(20000)    #load all test data
    dataset = dataset.batch(batch_size)

    return dataset

def make_model_base_SE(train_flag,name=None,init=None):
    input_data = tf.keras.Input(shape=(299, 299, 3),name='image')
    conv_base = inception_v3.InceptionV3(weights=init,
                                         include_top=False, input_shape=(299, 299, 3))(input_data)

    avepool1 = layers.GlobalAveragePooling2D()(conv_base)
    den1 = layers.Dense(512)(avepool1)
    drop1 = layers.Dropout(0.5)(den1)
    relu1 = layers.Activation('relu')(drop1)
    den2 = layers.Dense(2048)(relu1)
    drop2 = layers.Dropout(0.5)(den2)
    sig2 = layers.Activation('sigmoid')(drop2)
    expand1 = layers.Lambda(lambda x: tf.keras.backend.expand_dims(x, 1))(sig2)
    expand2 = layers.Lambda(lambda x: tf.keras.backend.expand_dims(x, 1))(expand1)
    tile1 = layers.Lambda(lambda x: tf.keras.backend.tile(x, [1, 8, 8, 1]))(expand2)
    mul2 = layers.multiply([conv_base, tile1])
    maxpool1 = layers.GlobalAveragePooling2D()(mul2)
    output = layers.Dense(3, activation='softmax')(maxpool1)
    model = models.Model(input_data, output,name=name)
    model.trainable=train_flag

    return model

def meancher_model():
    input_data = tf.keras.Input(shape=(299, 299, 3), name='image')
    teacher=make_model_base_SE(train_flag=False,name='teacher',init='imagenet')
    student=make_model_base_SE(train_flag=True,name='student',init='imagenet')

    t_output=teacher(input_data)
    s_output=student(input_data)

    output=tf.keras.layers.Concatenate(axis=-1)([t_output,s_output])
    model=models.Model(input_data,output)

    return model

def make_model_base(train_flag,name=None):
    input_data = tf.keras.Input(shape=(299, 299, 3),name='image')
    conv_base = inception_v3.InceptionV3(weights='imagenet',
                                         include_top=False, input_shape=(299, 299, 3))(input_data)

    avepool1 = layers.GlobalAveragePooling2D()(conv_base)
    den1 = layers.Dense(512)(avepool1)
    drop1 = layers.Dropout(0.5)(den1)
    relu1 = layers.Activation('relu')(drop1)
    den2 = layers.Dense(2048)(relu1)
    drop2 = layers.Dropout(0.5)(den2)
    sig2 = layers.Activation('sigmoid')(drop2)
    expand1 = layers.Lambda(lambda x: tf.keras.backend.expand_dims(x, 1))(sig2)
    expand2 = layers.Lambda(lambda x: tf.keras.backend.expand_dims(x, 1))(expand1)
    tile1 = layers.Lambda(lambda x: tf.keras.backend.tile(x, [1, 8, 8, 1]))(expand2)
    mul2 = layers.multiply([conv_base, tile1])
    maxpool1 = layers.GlobalAveragePooling2D()(mul2)
    output = layers.Dense(2, activation='softmax')(maxpool1)
    model = models.Model(input_data, output,name=name)
    model.trainable=train_flag

    return model
