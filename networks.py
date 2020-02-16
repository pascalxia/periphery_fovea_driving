# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 15:28:23 2017

@author: pasca
"""

import keras.layers as layers
import tensorflow as tf
import numpy as np
import keras.layers.wrappers as wps
from my_alexnet import AlexNet
from keras.layers.normalization import BatchNormalization

import pdb



GAUSSIAN_KERNEL_SIZE = 15
EPSILON = 1e-12

    
def pure_alex_encoder(target_input_size=None, no_pool5=False):  #target_input_size=[313,537] for peripheral and [185, 185] for foveal
    def feature_net(input_tensor):
        if target_input_size is not None:
            input_tensor = tf.image.resize_bilinear(input_tensor, target_input_size) 
        feature_map = AlexNet(input_tensor, no_pool5)
        return feature_map
    return feature_net
    
def thick_conv_lstm_readout_net(feature_map_in_seqs, feature_map_size, drop_rate, gaze_prior=None, output_embedding=False):
    batch_size = tf.shape(feature_map_in_seqs)[0]
    n_step = tf.shape(feature_map_in_seqs)[1]
    n_channel = int(feature_map_in_seqs.get_shape()[4])
    feature_map = tf.reshape(feature_map_in_seqs,  
                             [batch_size*n_step, feature_map_size[0], 
                              feature_map_size[1], n_channel])
    
    x = layers.Conv2D(16, (1, 1), activation='relu', name='readout_conv1')(feature_map)
    x = layers.BatchNormalization()(x)
    x = layers.core.Dropout(drop_rate)(x)
    x = layers.Conv2D(32, (1, 1), activation='relu', name='readout_conv2')(x)
    x = layers.BatchNormalization()(x)
    x = layers.core.Dropout(drop_rate)(x)
    x = layers.Conv2D(8, (1, 1), activation='relu', name='readout_conv3')(x)
    x = layers.BatchNormalization()(x)
    
    # reshape into temporal sequence
    temp_shape = x.get_shape()[1:4]
    temp_shape = [int(s) for s in temp_shape]
    x = tf.reshape(x, [batch_size, n_step, temp_shape[0], temp_shape[1], temp_shape[2]])
    
    n_channel = 15
    
    initial_c = layers.Conv2D(n_channel, (3, 3), activation='tanh', padding='same')(layers.core.Dropout(drop_rate)(x[:, 0]))
    initial_c = layers.core.Dropout(drop_rate)(initial_c)
    initial_h = layers.Conv2D(n_channel, (3, 3), activation='tanh', padding='same')(layers.core.Dropout(drop_rate)(x[:, 0]))
    initial_h = layers.core.Dropout(drop_rate)(initial_h)
    
    conv_lstm = layers.ConvLSTM2D(filters=n_channel,
                                  kernel_size=(3,3),
                                  strides=(1,1),
                                  padding='same', 
                                  dropout=drop_rate, 
                                  recurrent_dropout=drop_rate,
                                  return_sequences=True)
    x = conv_lstm([x, initial_c, initial_h])
    x = wps.TimeDistributed(layers.Conv2D(n_channel, (1, 1), activation='linear'))(x)
    x = tf.reshape(x, [batch_size*n_step, 
                       feature_map_size[0], feature_map_size[1], n_channel])
    x = layers.BatchNormalization()(x)
    embed = x
    
    x = layers.Conv2D(1, (1, 1), activation='linear')(x)
    
    x = tf.reshape(x, [batch_size*n_step, 
                       feature_map_size[0], feature_map_size[1], 1])
    raw_logits = tf.reshape(x, [-1, feature_map_size[0]*feature_map_size[1]])
    
    logits = tf.reshape(x, [-1, feature_map_size[0]*feature_map_size[1]])
    
    #gaze prior map
    if gaze_prior is not None:
        #predicted annotation before adding prior
        pre_prior_logits = logits

        gaze_prior = np.maximum(gaze_prior, EPSILON*np.ones(gaze_prior.shape))
        gaze_prior = gaze_prior.astype(np.float32)
        log_prior = np.log(gaze_prior)
        log_prior_1d = np.reshape(log_prior, (1, -1))
        log_prior_unit_tensor = tf.constant(log_prior_1d)
        log_prior_tensor = tf.matmul(tf.ones((tf.shape(pre_prior_logits)[0],1)), log_prior_unit_tensor)
        log_prior_tensor = tf.reshape(log_prior_tensor, 
                                      [-1, feature_map_size[0]*feature_map_size[1]])
        logits = tf.add(pre_prior_logits, log_prior_tensor)
    
    if output_embedding:
        return logits, embed, raw_logits
    
    if gaze_prior is None:
        return logits
    else:
        return logits, pre_prior_logits

def peripheral_readout_net(feature_map_in_seqs, feature_map_size, drop_rate):
    batch_size = tf.shape(feature_map_in_seqs)[0]
    n_step = tf.shape(feature_map_in_seqs)[1]
    n_channel = int(feature_map_in_seqs.get_shape()[4])
    feature_map = tf.reshape(feature_map_in_seqs,  
                             [batch_size*n_step, feature_map_size[0], 
                              feature_map_size[1], n_channel])
    
    x = layers.Conv2D(16, (1, 1), activation='relu', name='readout_conv1')(feature_map)
    x = layers.BatchNormalization()(x)
    x = layers.core.Dropout(drop_rate)(x)
    x = layers.Conv2D(32, (1, 1), activation='relu', name='readout_conv2')(x)
    x = layers.BatchNormalization()(x)
    x = layers.core.Dropout(drop_rate)(x)
    x = layers.Conv2D(8, (1, 1), activation='relu', name='readout_conv3')(x)
    x = layers.BatchNormalization()(x)
    
    # reshape into temporal sequence
    x = tf.reshape(x, [batch_size, n_step, feature_map_size[0], feature_map_size[1], 8])
    
    return x
    
def foveal_readout_net(feature_maps, drop_rate):
    #feature_maps shape: [batch_size*n_steps*N_FOVEAE, 5, 5, 256]
    x = layers.Conv2D(16, (3, 3), activation='relu', name='readout_conv1')(feature_maps)
    x = layers.BatchNormalization()(x)
    x = layers.core.Dropout(drop_rate)(x) # shape: [batch_size*n_steps*N_FOVEAE, 3, 3, 16]
    x = layers.Conv2D(32, (1, 1), activation='relu', name='readout_conv2')(x)
    x = layers.BatchNormalization()(x)
    x = layers.core.Dropout(drop_rate)(x) # shape: [batch_size*n_steps*N_FOVEAE, 3, 3, 32]
    x = layers.Conv2D(8, (1, 1), activation='relu', name='readout_conv3')(x)
    x = layers.BatchNormalization()(x) # shape: [batch_size*n_steps*N_FOVEAE, 3, 3, 8]
    return x
    
def conv_lstm_planner(peripheral_feature_map_seqs, foveal_feature_seqs, drop_rate):
    # combine feature maps
    if foveal_feature_seqs is None:
        feature_map_seqs = peripheral_feature_map_seqs
    elif peripheral_feature_map_seqs is None:
        feature_map_seqs = foveal_feature_seqs
    else:
        feature_map_seqs = tf.concat([peripheral_feature_map_seqs, foveal_feature_seqs], axis=-1)
    # get the shape
    batch_size = tf.shape(feature_map_seqs)[0]
    n_step = tf.shape(feature_map_seqs)[1]
    temp_shape = feature_map_seqs.get_shape()[2:5]
    temp_shape = [int(s) for s in temp_shape]
    feature_map_size = temp_shape[0:2]
    n_channel = temp_shape[2]
    
    conv_lstm = layers.ConvLSTM2D(filters=5,
                                  kernel_size=(3,3),
                                  strides=(1,1),
                                  padding='same', 
                                  dropout=drop_rate, 
                                  recurrent_dropout=drop_rate,
                                  return_sequences=True)
        
    initial_c = layers.Conv2D(5, (3, 3), activation='tanh', padding='same')(layers.core.Dropout(drop_rate)(feature_map_seqs[:, 0]))
    initial_c = layers.core.Dropout(drop_rate)(initial_c)
    initial_h = layers.Conv2D(5, (3, 3), activation='tanh', padding='same')(layers.core.Dropout(drop_rate)(feature_map_seqs[:, 0]))
    initial_h = layers.core.Dropout(drop_rate)(initial_h)
    x = conv_lstm([feature_map_seqs, initial_c, initial_h])
    
    # track weights
    kernel_weights = conv_lstm.weights[0] # shape is [3, 3, 8+8, 5*4]
    if peripheral_feature_map_seqs is None:
        peripheral_weights = None
        peripheral_n_channels = 0
    else:
        peripheral_n_channels = tf.shape(peripheral_feature_map_seqs)[-1]
        peripheral_weights = kernel_weights[:, :, 0:peripheral_n_channels, :]
        
    if foveal_feature_seqs is None:
        foveal_weights = None
    else:
        foveal_weights = kernel_weights[:, :, peripheral_n_channels:, :]
    
    x = tf.reshape(x, [batch_size*n_step, 
                       feature_map_size[0], feature_map_size[1], 5])
    x = layers.BatchNormalization()(x)
    
    temp_shape = x.get_shape()[1:4]
    temp_shape = [int(s) for s in temp_shape]
    x = tf.reshape(x, [batch_size, n_step, temp_shape[0]*temp_shape[1]*temp_shape[2]])
    
    x = wps.TimeDistributed(layers.Dense(units=512, activation='linear'))(x)

    logits = tf.reshape(x, [batch_size*n_step, 512])
    
    
    return logits, peripheral_weights, foveal_weights
    
    
def controller(flat_logits, drop_rate=0):
    x = flat_logits
    x = layers.Dense(units=100, activation='relu')(x)
    x = layers.Dense(units=50, activation='relu')(x)
    x = layers.Dense(units=10, activation='relu')(x)
    flat_output_speeds = layers.Dense(units=1, activation='relu')(x)
    return flat_output_speeds


