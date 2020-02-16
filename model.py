import tensorflow as tf 
import networks
from matplotlib_summary_op_factory import MatplotlibSummaryOpFactory
import os
import augment_images
import keras.layers as layers
import numpy as np

import pdb

N_FOVEAE = 2
TIME_INTERVAL = 0.1  # in seconds



# Define the model here
def model_fn(features, labels, mode, params):
    cameras = features['cameras']
    batch_size_tensor = tf.shape(cameras)[0]
    n_steps_tensor = tf.shape(cameras)[1]
    camera_input = tf.reshape(cameras, 
                            [-1, params['camera_size'][0], params['camera_size'][1], 3])
    peripheral_cameras = features['low_res_cameras']
    peripheral_cameras = tf.reshape(peripheral_cameras, 
                                    [-1, params['small_camera_size'][0], params['small_camera_size'][1], 3])

                            
    if params['foveal_only'] and params['attention_model_dir'] is None:
        # No need to make feature maps for the peripheral input
        feature_map_size = params['feature_map_size']
    else:
    # Make peripheral input
        # if params['augment_data']:
        #     peripheral_cameras, relative_translation = augment_images.augment_images(peripheral_cameras)
        peripheral_cameras = tf.cast(peripheral_cameras, tf.float32)
        peripheral_input = peripheral_cameras - [123.68, 116.79, 103.939]
    
        if params['premade_features']:
            feature_map_in_seqs = features['premade_features']
            feature_map_size = feature_map_in_seqs.get_shape().as_list()[2:4]
            n_channel = feature_map_in_seqs.get_shape().as_list()[4]
            feature_maps = tf.reshape(feature_map_in_seqs, 
                                      [batch_size_tensor*n_steps_tensor, feature_map_size[0], feature_map_size[1],
                                       n_channel])
        # Encode peripheral input
        with tf.variable_scope("p_encoder"):
            readout_network = networks.pure_alex_encoder(no_pool5=True)
            feature_maps = readout_network(peripheral_input) # shape: [batch_size*n_steps, 3, 7, 256]
        feature_maps = layers.UpSampling2D(size=(3, 2))(feature_maps) # shape: [batch_size*n_steps, 9, 14, 256]
        feature_maps = tf.concat([
            feature_maps[:, :, 0:1, :],
            feature_maps,
            feature_maps[:, :, -1:, :],
        ], axis=2) # shape: [batch_size*n_steps, 9, 16, 256]
        # reshape to sequences
        feature_map_size = feature_maps.get_shape().as_list()[1:3]
        n_channel = feature_maps.get_shape().as_list()[3]
        feature_map_in_seqs = tf.reshape(feature_maps,
                                         [batch_size_tensor, n_steps_tensor,
                                          feature_map_size[0], feature_map_size[1],
                                          n_channel])
                                              
            
                                          
    

    
    
    
    # Do not use foveal vision
    if not params['use_foveal']:
        f_feature_seqs = None
    # Use foveal vision and encode foveal input
    else:
        # Determine fovea locations
        h, w = feature_map_size[0], feature_map_size[1]
        if params['attention_model_dir'] is None and not params['premade_attention_maps']:
            # Use random foveal locations
            if params['random_fovea']:
                fovea_inds = tf.random_uniform(
                    [batch_size_tensor*n_steps_tensor*N_FOVEAE, ], 
                    minval=0, 
                    maxval=h*w, 
                    dtype=tf.int32
                )
            # Use center foveal locations
            else:
                fovea_inds = tf.tile([70, 73], [batch_size_tensor*n_steps_tensor, ])
                predicted_gazemaps = tf.sparse_to_dense(
                    [[0, 0, 4, 6], [0, 0, 4,9]],
                    [1, 1, 9, 16],
                    [0.5, 0.5],
                    default_value=0,
                    validate_indices=True
                )
                predicted_gazemaps = tf.tile(predicted_gazemaps, [batch_size_tensor, n_steps_tensor, 1, 1])
            
        # Predict attention
        else:
            if params['premade_attention_maps']:
                predicted_gazemaps = tf.reshape(features['gaze_maps'], [batch_size_tensor*n_steps_tensor, h, w, 1])
                predicted_gazemaps = tf.cast(predicted_gazemaps, tf.float32)
                predicted_gazemaps = predicted_gazemaps + 0.01 # to avoid log(0) when calculate logits
                predicted_gazemaps = predicted_gazemaps / tf.reduce_sum(predicted_gazemaps, axis=[1,2,3], keepdims=True)
                attention_ps = tf.reshape(predicted_gazemaps, [batch_size_tensor*n_steps_tensor, h*w])
                attention_logits = tf.log(attention_ps)
            else:
                with tf.variable_scope("readout"):
                    attention_readout_net = networks.thick_conv_lstm_readout_net
                    attention_logits = attention_readout_net(feature_map_in_seqs, 
                                               feature_map_size=feature_map_size, 
                                               drop_rate=0)
                # Determine which checkpoint to restore
                best_ckpt_dir = os.path.join(params['attention_model_dir'], 'best_ckpt')
                ckpt_name = [f.split('.index')[0] for f in os.listdir(best_ckpt_dir) if f.endswith('.index')][0]
                ckpt_path = os.path.join(best_ckpt_dir, ckpt_name)
                
                variables_to_restore = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='readout')
                assignment_map = {
                    v.name.split(':')[0]: v for v in variables_to_restore
                }
                tf.train.init_from_checkpoint(ckpt_path, assignment_map)
                # Generate gaze maps
                attention_ps = tf.nn.softmax(attention_logits)
                predicted_gazemaps = tf.reshape(attention_ps, [-1,]+feature_map_size+[1])
              
            # Sample from the predicted distribution
            if params['sample_fovea'] is True or params['attention_logit_factor'] is not None:
                fovea_inds = tf.multinomial(attention_logits*params['attention_logit_factor'], N_FOVEAE)
            else:
                _, fovea_inds = tf.nn.top_k(attention_logits, k=N_FOVEAE, sorted=False)
            
            # Sort the indices for creating sparse tensor
            fovea_inds, _ = tf.nn.top_k(fovea_inds, k=N_FOVEAE)
            fovea_inds = tf.reverse(fovea_inds, axis=[-1,])
            fovea_inds = tf.reshape(fovea_inds, [-1,])
  
        # Make foveal input
        fovea_hs = tf.floordiv(fovea_inds, w)
        fovea_ws = tf.mod(fovea_inds, w)
        h0 = tf.cast(fovea_hs-1, tf.float32)/tf.cast(h, tf.float32)
        h1 = tf.cast(fovea_hs+2, tf.float32)/tf.cast(h, tf.float32)
        w0 = tf.cast(fovea_ws-1, tf.float32)/tf.cast(w, tf.float32)
        w1 = tf.cast(fovea_ws+2, tf.float32)/tf.cast(w, tf.float32)
        boxes = tf.transpose([h0, w0, h1, w1])
        box_inds = tf_repeat(tf.range(batch_size_tensor*n_steps_tensor), N_FOVEAE)
        crop_size = tf.cast([params['camera_size'][0]/h*3, params['camera_size'][1]/w*3], tf.int32)
        foveal_cameras = tf.image.crop_and_resize(
          camera_input,
          boxes,
          box_inds,
          crop_size
        )
        foveal_input = foveal_cameras - [123.68, 116.79, 103.939]
        # foveal_input.shape: [batch_size*N_FOVEAE, 80, 80, 3]
  
        # Encode foveal input
        with tf.variable_scope("f_encoder"):
            foveal_readout_network = networks.pure_alex_encoder(target_input_size=[185,185])
            f_feature_maps = foveal_readout_network(foveal_input) # shape: [batch_size*N_FOVEAE, 5, 5, 256]

        
    if params['foveal_only']:
        peripheral_feature_map_seqs = None
    else:
        # Process peripheral feature maps
        with tf.variable_scope("p_processer"):
            if params['readout'] == 'default':
                readout_net = networks.peripheral_readout_net
            peripheral_feature_map_seqs = readout_net(
                feature_map_in_seqs, 
                feature_map_size=feature_map_size, 
                drop_rate=0.2)                                              
            
                         
    if params['use_foveal']:
        # Process foveal feature maps
        with tf.variable_scope("f_processer"):
            foveal_feature_maps = networks.foveal_readout_net(f_feature_maps, drop_rate=0.2) 
            # shape: [batch_size*n_steps*N_FOVEAE, 3, 3, 8]
  
        # Embed foveal feature into global feature maps
        def unravel_index(indices, dims):
            unraveled = np.unravel_index(indices, dims)
            index_matrix = np.vstack(unraveled).transpose()
            return index_matrix.astype(np.int32)
        
        temp_shape = [int(s) for s in foveal_feature_maps.get_shape()[-3:]]
        patch_h = temp_shape[0]
        patch_w = temp_shape[1]
        n_channel = temp_shape[2]
        index_matrix_shape = [batch_size_tensor*n_steps_tensor, N_FOVEAE, patch_h, patch_w, n_channel]
        n_index = tf.reduce_prod(index_matrix_shape)
        basic_indices = tf.py_func(unravel_index, [tf.range(n_index), index_matrix_shape], tf.int32)
        basic_indices.set_shape([None, 5])
        
        fovea_inds = tf.reshape(fovea_inds, [batch_size_tensor*n_steps_tensor*N_FOVEAE,])
        fovea_top_left = tf.py_func(unravel_index, [fovea_inds, [h, w]], tf.int32) - [int((patch_h-1)/2), int((patch_w-1)/2)]
        fovea_top_left.set_shape([None, 2])
        # fovea_top_left's shape: [batch_size*n_steps*N_FOVEA, 2]
        index_adjustment = tf.tile(tf.expand_dims(fovea_top_left, axis=1), [1, patch_h*patch_w*n_channel, 1])
        # index_adjustment's shape: [batch_size*n_steps*N_FOVEA, patch_h*patch_w*n_channel, 2]
        index_adjustment = tf.reshape(index_adjustment, [n_index, 2])
        # index_adjustment's shape: [batch_size*n_steps*N_FOVEA*patch_h*patch_w*n_channel, 2]
        index_adjustment = tf.concat([
            tf.zeros([n_index, 2], tf.int32),
            index_adjustment,
            tf.zeros([n_index, 1], tf.int32),
        ], axis=1)
        indices = basic_indices + index_adjustment
        
        values = tf.reshape(foveal_feature_maps, [-1,])
        
        valid_mask = tf.reduce_all([
            tf.greater(indices[:,2], -1), 
            tf.less(indices[:,2], h),
            tf.greater(indices[:,3], -1), 
            tf.less(indices[:,3], w),
        ], axis=0)
        # valid_mask's shape: [batch_size*n_steps*N_FOVEA*patch_h*patch_w*n_channel,]
        indices = tf.boolean_mask(indices, valid_mask, axis=0)
        values = tf.boolean_mask(values, valid_mask, axis=0)
        
        output_shape = [batch_size_tensor*n_steps_tensor, N_FOVEAE, h, w, n_channel]
        output_shape_64 = tf.cast(output_shape, tf.int64)
        f_feature_seqs = tf.SparseTensor(tf.cast(indices, tf.int64), values, output_shape_64)
        f_feature_seqs = tf.sparse_add(tf.zeros(output_shape), f_feature_seqs)
        f_feature_seqs = tf.reduce_max(f_feature_seqs, axis=1)
        f_feature_seqs = tf.reshape(f_feature_seqs, [batch_size_tensor, n_steps_tensor, h, w, n_channel])
        
    
    with tf.variable_scope("planner"):
        flat_logits, peripheral_weights, foveal_weights = networks.conv_lstm_planner(peripheral_feature_map_seqs, f_feature_seqs, drop_rate=0.2)
        flat_output_speeds = networks.controller(flat_logits)
  
    # Get other inputs
    video_id = features['video_id']
    predicted_time_points = features['predicted_time_points']     
  
    # get prediction
    output_speeds = tf.reshape(flat_output_speeds, [batch_size_tensor, n_steps_tensor, 1])
    predictions = {
      'output_speeds': output_speeds,
    }
    
    if params['use_foveal'] and not params['random_fovea']:
        summed_abs_foveal_features = tf.expand_dims(tf.reduce_sum(tf.abs(f_feature_seqs), axis=-1), axis=-1) #[batch_size, n_steps, h, w, 1]
        fovea_maps = tf.cast(tf.greater(summed_abs_foveal_features, 0), tf.float32)   #[batch_size, n_steps, h, w, 1]
        predicted_gazemap_seqs = tf.reshape(predicted_gazemaps, [batch_size_tensor, n_steps_tensor, h, w, 1])
        fovea_ps = tf.reduce_sum(fovea_maps * predicted_gazemap_seqs)/tf.cast(batch_size_tensor*n_steps_tensor, tf.float32)
        fovea_map_sums = tf.reduce_sum(fovea_maps, axis=[2, 3, 4], keepdims=True)
        temporal_overlap = tf.reduce_sum(fovea_maps[:, :-1] * fovea_maps[:, 1:] / fovea_map_sums[:, :-1])/tf.cast(batch_size_tensor*n_steps_tensor, tf.float32) #[batch_size, n_steps, h, w, 1]
  
    # If doing prediction, return here with prediction
    if mode == tf.estimator.ModeKeys.PREDICT:
        if 'output_details' in params and params['output_details'] == True:
            predictions['speed'] = features['speed']
            if params['use_foveal'] and not params['random_fovea']:
                predictions['likelihood'] = tf.tile([fovea_ps,], [batch_size_tensor,])
                predictions['overlap'] = tf.tile([temporal_overlap,], [batch_size_tensor,])
            else:
                predictions['likelihood'] = tf.tile([-1,], [batch_size_tensor,])
                predictions['overlap'] = tf.tile([-1,], [batch_size_tensor,])
            if params.get('visualization', False):
                predictions['cameras'] = features['cameras']
                predictions['low_res_cameras'] = features['low_res_cameras']
                predictions['gaze_maps'] = predicted_gazemap_seqs
                predictions['fovea_hs'] = tf.reshape(fovea_hs, [batch_size_tensor, n_steps_tensor, 2])
                predictions['fovea_ws'] = tf.reshape(fovea_ws, [batch_size_tensor, n_steps_tensor, 2])
        predictions['video_id'] = video_id
        predictions['predicted_time_points'] = predicted_time_points
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

            
  
    # set up loss
    # labels' shape: [batch_size, n_steps, 2]
    speeds = tf.norm(labels, 'euclidean', axis=2, keepdims=True) # speeds' shape: [batch_size, n_steps, 1] 
    
    assert_op = tf.assert_equal(
        tf.shape(speeds), tf.shape(output_speeds),
        data=[video_id, predicted_time_points, 
              speeds, output_speeds],
        summarize=1000,
        message="labels and outputs don't match in shape"
    )
    with tf.control_dependencies([assert_op]):
        speed_loss = tf.losses.absolute_difference(
            speeds, output_speeds, 
            weights=tf.expand_dims(features['weights'], axis=-1), # so that padded frames won't influence the loss because weights are padded to zeros
            reduction=tf.losses.Reduction.MEAN)
            
        weights = tf.expand_dims(features['weights'], axis=-1)  # shape: [batch_size, n_steps, 1]
        errors = output_speeds - speeds  # shape: [batch_size, n_steps, 1]
        error_derivatives = (errors[:, 1:, :] - errors[:, :-1, :])/TIME_INTERVAL  # shape: [batch_size, n_steps-1, 1]
        stability_loss = tf.reduce_mean(tf.square(error_derivatives) * weights[:, 1:, :])
        
        loss = speed_loss + params['stability_loss_weight']*stability_loss
  
    # set up training
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=params['learning_rate'])
        processer_variables = (
            tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='p_processer') +
            tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='f_processer'))
        planner_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 
                                            scope='planner')
        variables_to_train = processer_variables + planner_variables                       
        train_op = optimizer.minimize(loss, tf.train.get_or_create_global_step(),
                                  var_list=variables_to_train)
    else:
        train_op = None
    
    # Track gradients
    if mode == tf.estimator.ModeKeys.TRAIN:
        gradient_dict = {}
        gradient_output = optimizer.compute_gradients(
            loss,
            var_list=variables_to_train,
        )
        for gradient, tensor in gradient_output:
            gradient_dict[tensor.name] = gradient
        var_to_track = 'f_processer/readout_conv1/kernel:0'
        fovea_gradient = gradient_dict.get(var_to_track, None)
        var_to_track = 'planner/dense_4/kernel:0'
        speed_gradient = gradient_dict.get(var_to_track, None)
    else:
        fovea_gradient = None
        speed_gradient = None
    
    # set up metrics
    # Calculate KL-divergence
    mae = tf.metrics.mean_absolute_error(speeds, output_speeds)
    rmse = tf.metrics.root_mean_squared_error(speeds, output_speeds)
    metrics = {
        'mae': mae,
        'rmse': rmse,
    }
    
    if 'skip_summary' in params and params['skip_summary'] == True:
        # Skip summary and return the model without summary ops
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            loss=loss,
            train_op=train_op,
            eval_metric_ops=metrics)
  
    # set up summaries
    quick_summaries = []
    quick_summaries.append(tf.summary.scalar('mae', mae[1]))
    quick_summaries.append(tf.summary.scalar('rmse', rmse[1]))
    quick_summaries.append(tf.summary.scalar('loss', loss))
    quick_summaries.append(tf.summary.scalar('speed_loss', speed_loss))
    quick_summaries.append(tf.summary.scalar('stability_loss', stability_loss))
    if fovea_gradient is not None:
        quick_summaries.append(tf.summary.scalar('fovea_processor_gradient', tf.reduce_mean(tf.abs(fovea_gradient))))
    if speed_gradient is not None:
        quick_summaries.append(tf.summary.scalar('speed_processor_gradient', tf.reduce_mean(tf.abs(speed_gradient))))
    
    quick_summary_op = tf.summary.merge(quick_summaries, name='quick_summary')
    quick_summary_hook = tf.train.SummarySaverHook(
        save_secs = params['quick_summary_period'],
        output_dir=params['model_dir'],
        summary_op=quick_summary_op
    )
        
    # slow summary
    def plot_actions(ax, speeds):
        # f is the matplotlib figure
        # digit is a numpy version of the argument passed to matplotlib_summary
        ax.plot(speeds)
        ax.set_title("Speed-time curve")

    matplotlib_summary_op = MatplotlibSummaryOpFactory()
    
    slow_summaries = []
    slow_summaries.append(tf.summary.image('peripheral_vision', peripheral_cameras, max_outputs=2))
    if params['use_foveal']:
        slow_summaries.append(tf.summary.image('foveal_vision', foveal_cameras, max_outputs=2))
        slow_summaries.append(tf.summary.histogram("foveal_weights", foveal_weights))
        slow_summaries.append(tf.summary.histogram("foveal_features", foveal_feature_maps))
        if not params['random_fovea']:
            slow_summaries.append(
                tf.summary.image(
                    'sparse_foveal_feature_maps', 
                    tf.reshape(summed_abs_foveal_features, [batch_size_tensor*n_steps_tensor, h, w, 1]), 
                    max_outputs=2))
        if params['attention_model_dir'] is not None:
            slow_summaries.append(tf.summary.image('predicted_gazemaps', predicted_gazemaps, max_outputs=2))
            slow_summaries.append(tf.summary.scalar('fovea_ps', fovea_ps))
            slow_summaries.append(tf.summary.scalar('temporal_overlap', temporal_overlap))
    slow_summaries.append(matplotlib_summary_op(plot_actions, speeds[0, :, 0], name="groundtruth_speeds"))
    slow_summaries.append(tf.summary.histogram("groundtruth_histogram", speeds))
    if not params['foveal_only']:
        slow_summaries.append(tf.summary.histogram("peripheral_input_histogram", peripheral_input))
        slow_summaries.append(tf.summary.histogram("peripheral_weights", peripheral_weights))
    slow_summaries.append(matplotlib_summary_op(plot_actions, output_speeds[0, :, 0], name="predicted_speeds"))
    slow_summaries.append(matplotlib_summary_op(plot_actions, error_derivatives[0, :, 0], name="error_derivatives"))
    slow_summaries.append(tf.summary.scalar('stability_loss_weight', params['stability_loss_weight']))
    slow_summary_op = tf.summary.merge(slow_summaries, name='slow_summary')
    slow_summary_hook = tf.train.SummarySaverHook(
        save_secs = params['slow_summary_period'],
        output_dir=params['model_dir'],
        summary_op=slow_summary_op
    )
  
    eval_summary_hook = tf.train.SummarySaverHook(
        save_secs = params['slow_summary_period'],
        output_dir=os.path.join(params['model_dir'], 'eval'),
        summary_op=slow_summary_op
    )
  
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=metrics,
        training_hooks=[quick_summary_hook, slow_summary_hook],
        evaluation_hooks=[eval_summary_hook]
    )
    
    
def tf_repeat(a_tensor, repeat):
    a_tensor = tf.reshape(a_tensor, [-1, 1])    # Convert to a n x 1 matrix.
    a_tensor = tf.tile(a_tensor, [1, repeat])  # Create multiple columns.
    a_tensor = tf.reshape(a_tensor, [-1])       # Convert back to a vector.
    return a_tensor
