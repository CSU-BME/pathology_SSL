import tensorflow.compat.v1 as tf
from absl import logging
from inception_preprocessing import preprocess_image
from tensorflow.compat.v1.keras.applications import inception_v3
from tensorflow.compat.v1.keras import models
from tensorflow.compat.v1.keras import layers
import tensorflow.compat.v1.keras.backend as KTF
from sklearn import metrics

num_classes=2     #2 classs

def load_dataset(filename):
  """Reads a file in the `.tfrecord` format.

  Args:
    filename: Name of the file containing `tf.train.Example` objects.

  Returns:
    An instance of `tf.data.TFRecordDataset` containing the `tf.train.Example`
    objects.
  """
  return tf.data.TFRecordDataset([filename])

def parse_example(example_proto):
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
            features[key] = preprocess_image(features[key], 299, 299,True)

    features.pop('id')
    features['label'] = tf.one_hot(features['label'], num_classes)
    labels = features.pop('label')

    return features, labels

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
            features[key] = preprocess_image(features[key], 299, 299)

    features.pop('id')
    features['label'] = tf.one_hot(features['label'], num_classes)
    labels = features.pop('label')

    return features, labels


def make_dataset(file_path, batch_size,training=False):
    """Creates a `tf.data.Dataset`."""
    # If the dataset is sharded, the following code may be required:
    # filenames = tf.data.Dataset.list_files(file_path, shuffle=True)
    # dataset = filenames.interleave(load_dataset, cycle_length=1)
    dataset = load_dataset(file_path)
    if training:
        dataset = dataset.shuffle(200000).repeat()
        dataset = dataset.map(parse_example)
    else:
        dataset = dataset.shuffle(50000)
        dataset = dataset.map(parse_example_val)

    dataset = dataset.batch(batch_size)
    return dataset


def make_datasets(train_data_path, val_data_path, test_data_path,batch_size=32):
    """Returns training and test data as a pair of `tf.data.Dataset` instances."""
    return make_dataset(train_data_path, batch_size,True), make_dataset(val_data_path,batch_size), \
           make_dataset(test_data_path,batch_size)

def log_metrics(model_desc, eval_metrics):
  """Logs evaluation metrics at `logging.INFO` level.

  Args:
    model_desc: A description of the model.
    eval_metrics: A dictionary mapping metric names to corresponding values. It
      must contain the loss and accuracy metrics.
  """
  logging.info('\n')
  logging.info('Eval accuracy for %s: %s', model_desc, eval_metrics['acc'])
  logging.info('Eval loss for %s: %s', model_desc, eval_metrics['loss'])

def make_model_base(train_flag=False,name=None,init=None):
    input_data = tf.keras.Input(shape=(299, 299, 3),name='image')
    conv_base = inception_v3.InceptionV3(weights=init,
                                         include_top=False, input_shape=(299, 299, 3))(input_data)

    maxpool1 = layers.GlobalAveragePooling2D()(conv_base)
    output = layers.Dense(num_classes, activation='softmax')(maxpool1)
    model = models.Model(input_data, output,name=name)
    model.trainable=train_flag

    return model

def auc(y_true,y_pred):
    y_pred = tf.cast(tf.argmax(y_pred, axis=-1), dtype=tf.float32)   #two classes
    y_true = tf.cast(tf.argmax(y_true, axis=-1), dtype=tf.float32)
    y_true_f = KTF.flatten(y_true)
    y_pred_f = KTF.flatten(y_pred)

    return tf.numpy_function(metrics.roc_auc_score, (y_true_f, y_pred_f), tf.double)

def make_test_dataset(file_path, batch_size):
    """Creates a `tf.data.Dataset`."""
    dataset = load_dataset(file_path)
    dataset = dataset.map(parse_example_val)
    dataset = dataset.shuffle(32768)  # load all test data
    dataset = dataset.batch(batch_size)
    return dataset