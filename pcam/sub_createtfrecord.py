import tensorflow.compat.v1 as tf
import glob
import os
import cv2
import sys

input_path='/media/disk3/pcam-semi-up/subs-pcam-0.01'
output_path='/media/disk3/pcam-semi-up/subs-pcam-0.01'
dirs=['0','1','2']     #possible labels, 2 is no-labels

def int64_feature(values):
  """Returns a TF-Feature of int64s.

  Args:
    values: A scalar or list of values.

  Returns:
    a TF-Feature.
  """
  if not isinstance(values, (tuple, list)):
    values = [values]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

def bytes_feature(values):
  """Returns a TF-Feature of bytes.

  Args:
    values: A string.

  Returns:
    a TF-Feature.
  """
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))

def create_tfrecord(input_path,writer):
   for one_dir in dirs:
      print('process: '+one_dir+'\n')
      files = glob.glob(os.path.join(input_path, one_dir) + '/*.*')

      i = 0
      for one_file in files:
          i += 1
          sys.stdout.write('\r>> Converting image %d/%d on dir %s' % (
              i, len(files), one_dir))
          sys.stdout.flush()

          image_data = tf.gfile.FastGFile(one_file, 'rb').read()
          label = int(one_dir)
          basename = os.path.basename(one_file)
          id = basename[0:basename.rfind('.')].encode()

          features = {
            'id': bytes_feature(id),
            'image': bytes_feature(image_data),
            'label': int64_feature(label),
          }

          example_features = tf.train.Example(
              features=tf.train.Features(feature=features))

          writer.write(example_features.SerializeToString())

datasets=[a for a in os.listdir(input_path) if os.path.isdir(os.path.join(input_path,a))]
for dataset in datasets:
    datapath=os.path.join(input_path,dataset)

    path = os.path.join(datapath, 'train')
    output_file=os.path.join(output_path,'train-sub'+dataset+'.tfr')
    writer = tf.io.TFRecordWriter(output_file)
    create_tfrecord(path,writer)

    path = os.path.join(datapath, 'test')
    output_file = os.path.join(output_path, 'test-sub' + dataset + '.tfr')
    writer = tf.io.TFRecordWriter(output_file)
    create_tfrecord(path, writer)

    path = os.path.join(datapath, 'extra')
    output_file = os.path.join(output_path, 'extra-sub' + dataset + '.tfr')
    writer = tf.io.TFRecordWriter(output_file)
    create_tfrecord(path, writer)

    path = os.path.join(datapath, 'valid')
    output_file = os.path.join(output_path, 'val-sub' + dataset + '.tfr')
    writer = tf.io.TFRecordWriter(output_file)
    create_tfrecord(path, writer)
    writer.close()

    writer.close()
