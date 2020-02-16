import os
import tensorflow as tf
from nexar_large_speed import MyDataset
import augment_images
import pdb


SPEED_LIMIT_AS_STOP = 2


def get_sample_prob(example):
    sequence_feature_info = {
        'weights': tf.FixedLenSequenceFeature(shape=[], dtype=tf.float32)
    }
    _, sequence_features = tf.parse_single_sequence_example(example, 
        sequence_features=sequence_feature_info)
    sample_prob = tf.reduce_mean(sequence_features['weights'])
    return sample_prob


def oversample_classes(example):
    """
    Returns the number of copies of given example
    """
    sample_prob = get_sample_prob(example)
    # for sample_prob smaller than 1, we
    # want to return 1
    sample_prob = tf.maximum(sample_prob, 1) 
    # for low probability classes this number will be very large
    repeat_count = tf.floor(sample_prob)
    # sample_prob can be e.g 1.9 which means that there is still 90%
    # of change that we should return 2 instead of 1
    repeat_residual = sample_prob - repeat_count # a number between 0-1
    residual_acceptance = tf.less_equal(
                        tf.random_uniform([], dtype=tf.float32), repeat_residual
    )
    
    residual_acceptance = tf.cast(residual_acceptance, tf.int64)
    repeat_count = tf.cast(repeat_count, dtype=tf.int64)
    
    return repeat_count + residual_acceptance


def undersampling_filter(example):
    """
    Computes if given example is rejected or not.
    """
    sample_prob = tf.minimum(sample_prob, 1.0)
    
    acceptance = tf.less_equal(tf.random_uniform([], dtype=tf.float32), sample_prob)

    return acceptance


def input_fn(dataset, batch_size, n_steps, shuffle, include_labels, n_epochs, args, weight_data=False, augment_data=False):
  """Prepare data for training."""
  
  # get and shuffle tfrecords files
  if include_labels:
    if args.premade_attention_maps:
      files = tf.data.Dataset.list_files(os.path.join(args.data_dir, 'bddx_tfrecords_gazemaps', dataset,
        '*.tfrecords'))
    elif args.premade_features:
      files = tf.data.Dataset.list_files(os.path.join(args.data_dir, 'bddx_tfrecords_features', dataset,
        '*.tfrecords'))
    elif args.multiple_tfrecords:
      files = tf.data.Dataset.list_files(os.path.join(args.data_dir, 'tfrecords_segments', dataset,
        '*.tfrecords'))
    else:
      files = tf.data.Dataset.list_files(os.path.join(args.data_dir, 'tfrecords', dataset,
        '*.tfrecords'))
  else:
    files = tf.data.Dataset.list_files(os.path.join(args.data_dir, dataset, 'tfrecords',
      'cameras_*.tfrecords'))
  if shuffle:
    files = files.shuffle(buffer_size=100)
  
  # parellel interleave to get raw bytes
  dataset = files.apply(tf.contrib.data.parallel_interleave(
    tf.data.TFRecordDataset, cycle_length=args.num_parallel, block_length=batch_size))
  
  # if apply weighted sampling
  if weight_data:
    dataset = dataset.flat_map(
      lambda x: tf.data.Dataset.from_tensors(x).repeat(oversample_classes(x))
    )
    dataset = dataset.filter(undersampling_filter)
  
  # shuffle before parsing
  # if shuffle:
  #   dataset = dataset.shuffle(buffer_size=5*batch_size)
  
  # Parse data
  def _parse_function(example):
    # specify feature information
    feature_map = {
            'image/encoded': tf.VarLenFeature(dtype=tf.string),
            'image/low_res': tf.VarLenFeature(dtype=tf.string),
            'image/class/video_name': tf.FixedLenFeature([1], dtype=tf.string, default_value=''),
    }
    if args.premade_attention_maps:
      feature_map['image/gaze_maps'] = tf.VarLenFeature(dtype=tf.string)
    if args.premade_features:
      feature_map['image/premade_features'] = tf.VarLenFeature(dtype=tf.float32)
    if include_labels:
      feature_map['image/speeds'] = tf.VarLenFeature(dtype=tf.float32)
    if args.multiple_tfrecords:
      feature_map['time_points'] = tf.VarLenFeature(dtype=tf.int64)
    # parse the example
    features = tf.parse_single_example(example, feature_map)
    return features
  dataset = dataset.map(_parse_function, num_parallel_calls=args.num_parallel)
  
  # Check long enough to cover n_future_steps
  if args.check_long_enough:
      def _long_enough(sequence_features):
        if include_labels:
          length = tf.minimum(
            tf.shape(sequence_features["image/low_res"])[0],
            tf.cast(tf.shape(sequence_features["image/speeds"])[0]/2, tf.int32) #because speeds are flattened x,y components
          )
        else:
          length = tf.shape(sequence_features["image/low_res"])[0]
        return tf.less(args.n_future_steps, length)
      dataset = dataset.filter(_long_enough)
  
  # Post-process
  def _process_fn(features):
    name = features['image/class/video_name']
    encoded = features['image/encoded'].values
    low_res = features['image/low_res'].values
    if args.premade_attention_maps:
      gaze_maps = features['image/gaze_maps'].values
    if args.premade_features:
      premade_features = features['image/premade_features'].values
      premade_features = tf.reshape(premade_features, [-1,]+args.input_feature_map_size+[args.feature_map_channels,])
    if include_labels:
      speed = features['image/speeds'].values
      speed = tf.reshape(speed, [-1, 2])
    # Determine the length of the example
    if include_labels:
      length = tf.minimum(
        tf.shape(features["image/low_res"])[0],
        tf.cast(tf.shape(features["image/speeds"])[0]/2, tf.int32) #because speeds are flattened x,y components
      )
    else:
      length = tf.shape(features["image/low_res"])[0]
    if args.multiple_tfrecords:
      time_points = features['time_points'].values
    else:
      time_points = tf.range(length)
    # determine the start and end of the sequence to pass
    if n_steps is None:
      start = 0
      end = length - args.n_future_steps
    else:
      # If n_steps > entire sequence length - n_future_steps, use the whole sequence
      # otherwise, sample a subsequence
      # Define the sampling function that is potentially needed
      def _sample_subsequence():
        """
        sample the starting point (offset) according to the sampling weights of windows
        """
        if weight_data:
          cum_weights = tf.cumsum(weights[:-args.n_future_steps], axis=0)
          sample_prob = cum_weights[n_steps-1:] - tf.concat([[0,], cum_weights[:-n_steps]], axis=0)
          sample_prob = sample_prob / tf.reduce_sum(sample_prob)
          start = tf.multinomial(logits=tf.log([sample_prob,]), num_samples=1, output_dtype=tf.int32)[0, 0]
        else:
          start = tf.random_uniform(shape=[], minval=0,
                                    maxval=length - args.n_future_steps - n_steps + 1,
                                    dtype=tf.int32)
        end = start + n_steps
        return start, end
      start, end = tf.cond(tf.greater(n_steps, length - args.n_future_steps),
                           lambda: (0, length - args.n_future_steps),
                           _sample_subsequence)
    encoded = encoded[start:end]
    low_res = low_res[start:end]
    if args.premade_attention_maps:
      gaze_maps = gaze_maps[start:end]
    if args.premade_features:
      premade_features = premade_features[start:end]
    
    predicted_time_points = time_points[start+args.n_future_steps:end+args.n_future_steps]
    if include_labels:
      speed = speed[start+args.n_future_steps:end+args.n_future_steps]
    # post-process data
    # decode jpg's
    encoded = tf.map_fn(
      tf.image.decode_jpeg,
      encoded,
      dtype=tf.uint8,
      back_prop=False
    )
    low_res = tf.map_fn(
      tf.image.decode_jpeg,
      low_res,
      dtype=tf.uint8,
      back_prop=False
    )
    if args.premade_attention_maps:
      gaze_maps = tf.map_fn(
        tf.image.decode_jpeg,
        gaze_maps,
        dtype=tf.uint8,
        back_prop=False
      )
    if args.discrete_output:
        # process speed information
        #speed.set_shape([tf.shape(encoded)[0], 2])
        # Note also that stop_future_frames is reused for the turn
        def turn_future_smooth(speed, nfuture, speed_limit_as_stop):
            # this function takes in the speed and output a smooth future action map
            turn = MyDataset.turning_heuristics(speed, speed_limit_as_stop)
            #smoothed = MyDataset.future_smooth(turn, naction=4, nfuture=nfuture) # naction=4 because ignore slight turns
            return turn
        turn = tf.py_func(turn_future_smooth,
                          [speed, args.n_future_steps, SPEED_LIMIT_AS_STOP],
                          [tf.int32])[0]  #TODO(lowres)
        turn = tf.maximum(turn, 0) # turn -1 to 0 (uncertain -> go straight)
        turn = tf.one_hot(turn, depth=4)
    # handle sampling weights
    if not weight_data:
      weights = tf.ones(tf.shape(encoded)[0:1])
    else:
      weights = tf.tile(tf.reduce_mean(weights, axis=0, keepdims=True), [tf.shape(encoded)[0],])
    # return features
    features = {}
    features['cameras'] = encoded
    features['low_res_cameras'] = low_res
    features['video_id'] = name[0]
    features['predicted_time_points'] = predicted_time_points
    features['weights'] = weights
    if args.premade_attention_maps:
      features['gaze_maps'] = gaze_maps
    if args.premade_features:
      features['premade_features'] = premade_features
    if include_labels:
      features['speed'] = speed
      if args.discrete_output:
        features['actions'] = turn
    return features
  dataset = dataset.map(_process_fn, num_parallel_calls=args.num_parallel)
  
  # Image augmentation
  def _image_augmentation(features):
    if not augment_data:
      features['translation'] = tf.zeros([2,], dtype=tf.float32)
      return features
    else:
      features['low_res_cameras'], features['translation'] = augment_images.augment_images(
        features['low_res_cameras']
      )
      return features
  dataset = dataset.map(_image_augmentation, num_parallel_calls=args.num_parallel)
  
  # Generate labels
  def _generate_labels(features):
    if args.discrete_output:
      labels = tf.expand_dims(features['actions'], -1)
    else:
      labels = features['speed']
    return features, labels
  if include_labels:
    dataset = dataset.map(_generate_labels, num_parallel_calls=args.num_parallel)
  
  # Pad batched data
  padded_shapes = {'cameras': [None,]+args.camera_size+[3],
                   'low_res_cameras': [None,]+args.small_camera_size+[3],
                   'translation': [2,],
                   'video_id': [],
                   'predicted_time_points': [None,],
                   'weights': [None,],
                   'speed': [None, 2]}
  if args.premade_attention_maps:
    padded_shapes['gaze_maps'] = [None,]+args.gazemap_size+[1]
  if args.premade_features:
    padded_shapes['premade_features'] = [None,]+args.input_feature_map_size+[args.feature_map_channels]
  if include_labels:
    if args.discrete_output:
        padded_shapes['actions'] = [None, 4]
        padded_shapes = (padded_shapes, [None, 4])
    else:
        padded_shapes = (padded_shapes, [None, 2])
  if args.pad_batch:
    dataset = dataset.padded_batch(batch_size, padded_shapes=padded_shapes)
  else:
    dataset = dataset.batch(batch_size)
  
  # Prefetch and repeat
  dataset = dataset.prefetch(buffer_size=batch_size)
  dataset = dataset.repeat(n_epochs)
  
  return dataset
  
  
  
  
  

