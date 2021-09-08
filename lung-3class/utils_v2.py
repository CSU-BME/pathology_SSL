import tensorflow.compat.v1 as tf
from absl import logging
from inception_preprocessing import preprocess_image
#read data
num_classes=3+1     #3 classs and 1 unknown class

def load_dataset(filename):
  return tf.data.TFRecordDataset([filename])

def parse_example(example_proto):
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
            features[key]=tf.random_crop(features[key], (299, 299, 3))   #random crop patches for training.
            features[key]=preprocess_image(features[key], 299, 299,True)     #have argumentation, for train.

    features.pop('id')
    features['label'] = tf.one_hot(features['label'], num_classes)
    features['label']=tf.concat([features['label'],features['label']],axis=0)   #add false label
    labels = features.pop('label')

    return features, labels

def parse_example_val(example_proto):
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
            features[key]=tf.image.crop_to_bounding_box(features[key], 234, 234, 299, 299)
            features[key]=preprocess_image(features[key], 299, 299)    #no augmentation

    features.pop('id')
    features['label'] = tf.one_hot(features['label'], num_classes)
    features['label']=tf.concat([features['label'],features['label']],axis=0)   #add false label
    labels = features.pop('label')

    return features, labels

def parse_example_read(example_proto):
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
            features[key] = tf.image.crop_to_bounding_box(features[key], 234, 234, 299, 299)

    features.pop('id')
    features['label'] = tf.one_hot(features['label'], num_classes)
    features['label']=tf.concat([features['label'],features['label']],axis=0)   #add false label
    labels = features.pop('label')

    return features, labels

def make_dataset(file_path, batch_size,training,extra_path=None):
    dataset_extra=None
    dataset = load_dataset(file_path)

    if training==1:
        dataset = dataset.map(parse_example_read)
        dataset = dataset.shuffle(5000).repeat()

        if extra_path != None:
            dataset_extra = load_dataset(extra_path)
            dataset_extra = dataset_extra.map(parse_example_read)
            dataset_extra = dataset_extra.shuffle(5000).repeat()
            # dataset = tf.data.experimental.sample_from_datasets([dataset, dataset_extra], weights=[0.5, 0.5])
            dataset = tf.data.Dataset.zip((dataset, dataset_extra))

            dataset = dataset.flat_map(
                lambda pos, neg: tf.data.Dataset.from_tensors(pos).concatenate(
                    tf.data.Dataset.from_tensors(neg)))
            dataset = dataset.batch(batch_size)
    else:
        if training == 2:
            dataset = dataset.shuffle(3000).repeat()
            dataset = dataset.map(parse_example_read)  # for val while training
        if training == 3:
            dataset = dataset.shuffle(3000)
            dataset = dataset.map(parse_example_val)  # for test after training

        dataset = dataset.batch(batch_size)

    return dataset

def make_datasets(train_data_path, val_data_path, test_data_path,batch_size=32,extra_train_path=None):
    return make_dataset(train_data_path, batch_size, 1, extra_train_path), make_dataset(val_data_path, batch_size, 2), \
           make_dataset(test_data_path, batch_size, 3)

def log_metrics(model_desc, eval_metrics,key):
  logging.info('\n')
  logging.info('Eval accuracy for %s: %s', model_desc, eval_metrics[key])
  logging.info('Eval loss for %s: %s', model_desc, eval_metrics['loss'])

def make_test_dataset(file_path, batch_size):
    dataset = load_dataset(file_path)
    dataset = dataset.map(parse_example_val)
    dataset = dataset.shuffle(3000)
    dataset = dataset.batch(batch_size)
    return dataset

