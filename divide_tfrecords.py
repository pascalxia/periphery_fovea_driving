import argparse
import sys
import os

import tensorflow as tf 

import networks

import add_args
from keras import backend as K
import shutil
import numpy as np
import scipy.misc as misc
import math

import pdb

#MAX_LENGTH = 100


def input_fn(dataset, batch_size, n_epochs, args):
  """Prepare data for training."""
  
  files = tf.data.Dataset.list_files(os.path.join(args.data_dir, 'tfrecords', dataset,
    '*.tfrecords'))
  
  # parellel interleave to get raw bytes
  dataset = files.apply(tf.contrib.data.parallel_interleave(
    tf.data.TFRecordDataset, cycle_length=10, block_length=batch_size))
  
  # Parse data
  def _parse_function(example):
    # specify feature information
    feature_map = {
            'image/encoded': tf.VarLenFeature(dtype=tf.string),
            'image/low_res': tf.VarLenFeature(dtype=tf.string),
            'image/class/video_name': tf.FixedLenFeature([1], dtype=tf.string, default_value=''),
    }
    feature_map['image/speeds'] = tf.VarLenFeature(dtype=tf.float32)
    # parse the example
    features = tf.parse_single_example(example, feature_map)
    return features
  dataset = dataset.map(_parse_function, num_parallel_calls=10)
  
  # Post-process
  def _process_fn(features):
    video_id = features['image/class/video_name'][0]
    encoded = features['image/encoded'].values
    low_res = features['image/low_res'].values
    speed = features['image/speeds'].values
    length = tf.minimum(
      tf.shape(features["image/low_res"])[0],
      tf.cast(tf.shape(features["image/speeds"])[0]/2, tf.int32) #because speeds are flattened x,y components
    )
    time_points = tf.range(length)
    features = {}
    features['cameras'] = encoded
    features['low_res_cameras'] = low_res
    features['video_id'] = video_id
    features['speed'] = tf.reshape(speed, [-1, 2])
    features['time_points'] = time_points
    return features
  dataset = dataset.map(_process_fn, num_parallel_calls=10)
    
  # Prefetch and repeat
  dataset = dataset.prefetch(buffer_size=batch_size)
  dataset = dataset.repeat(n_epochs)
  
  return dataset

def _int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
    """Wrapper for inserting float features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def main(argv):
    parser = argparse.ArgumentParser()
    add_args.for_general(parser)
    add_args.for_inference(parser)
    add_args.for_evaluation(parser)
    add_args.for_feature(parser)
    add_args.for_lstm(parser)
    parser.add_argument('--max_length',
                            default=310,
                            type=int,
                            help="maximum length of one tfrecord")
    parser.add_argument('--data_subset',
                            default='test',
                            type=str,
                            help="the tfrecords of which subset to divide, i.e., training, validation or test")
    args = parser.parse_args()
    MAX_LENGTH = args.max_length
    
    if args.visible_gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpus
    
    sess = tf.Session()
    
    dataset = args.data_subset
    output_dir = args.data_dir + '/' + 'tfrecords_segments' + '/' + dataset
    if not os.path.isdir(output_dir):
      os.makedirs(output_dir)
    this_input_fn=lambda: input_fn(dataset,
      args.validation_batch_size,
      n_epochs=1, args=args)

    ds = this_input_fn()
    iterator = ds.make_one_shot_iterator()
    next_element = iterator.get_next()
    while True:
        try:
            res = sess.run(next_element)

            video_path = res['video_id'].decode("utf-8")
            video_id = video_path.split('/')[-1].split('.')[0]
            #premade_feature_dir = '/data/alexnet_features/' + dataset
            #premade_features = np.load(os.path.join(premade_feature_dir, video_id+'.npy'))
            
            length = len(res['cameras'])
            for i in range(math.ceil(float(length)/MAX_LENGTH)):
                startIdx = i*MAX_LENGTH
                endIdx = min( (i+1)*MAX_LENGTH, length )
                example = tf.train.Example(features=tf.train.Features(feature={
                    'image/class/video_name':_bytes_feature([video_path.encode('utf-8')]),
                    'image/encoded': _bytes_feature(res['cameras'][startIdx:endIdx]),
                    'image/low_res': _bytes_feature(res['low_res_cameras'][startIdx:endIdx]),
                    'image/speeds': _float_feature(res['speed'][startIdx:endIdx].ravel().tolist()), # ravel l*2 into list
                    'time_points': _int64_feature(res['time_points'][startIdx:endIdx].tolist()),
                    #'image/premade_features': _float_feature(premade_features[startIdx:endIdx].ravel().tolist()),
                }))

                writer = tf.python_io.TFRecordWriter(os.path.join(output_dir, video_id+'_'+str(i).zfill(2)+'.tfrecords'))
                writer.write(example.SerializeToString())
                writer.close()

            print(video_id)

        except tf.errors.OutOfRangeError:
            break
        
        
    # dataset = 'training'
    # output_dir = '/data/bddx_tfrecords_features/' + dataset
    # this_input_fn=lambda: input_fn(dataset,
    #   args.validation_batch_size, 
    #   n_epochs=1, args=args)
    #   
    # ds = this_input_fn()
    # iterator = ds.make_one_shot_iterator()
    # next_element = iterator.get_next()
    # while True:
    #     try:
    #         res = sess.run(next_element)
    #         
    #         video_path = res['video_id'].decode("utf-8")
    #         video_id = video_path.split('/')[-1].split('.')[0]
    #         premade_feature_dir = '/data/alexnet_features/' + dataset
    #         premade_features = np.load(os.path.join(premade_feature_dir, video_id+'.npy'))
    #         pdb.set_trace()
    #         
    #         example = tf.train.Example(features=tf.train.Features(feature={
    #             'image/class/video_name':_bytes_feature([video_path.encode('utf-8')]),
    #             'image/encoded': _bytes_feature(res['cameras']),
    #             'image/low_res': _bytes_feature(res['low_res_cameras']),
    #             'image/speeds': _float_feature(res['speed'].tolist()), # ravel l*2 into list
    #             'image/premade_features': _float_feature(premade_features.ravel().tolist()),
    #         }))
    #         
    #         writer = tf.python_io.TFRecordWriter(os.path.join(output_dir, video_id+'.tfrecords'))
    #         writer.write(example.SerializeToString())
    #         writer.close()
    #         
    #         print(video_id)
    #         
    #     except tf.errors.OutOfRangeError:
    #         break
  
  
  
  



if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  main(argv=sys.argv)
