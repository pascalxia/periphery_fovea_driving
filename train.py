
import argparse
import sys
import os

import tensorflow as tf 

import networks

import add_args
from keras import backend as K
import shutil
from input_data import input_fn
from model import model_fn

import pdb


def main(argv):
  
  parser = argparse.ArgumentParser()
  add_args.for_general(parser)
  add_args.for_inference(parser)
  add_args.for_feature(parser)
  add_args.for_training(parser)
  add_args.for_lstm(parser)
  args = parser.parse_args()
  
  config = tf.estimator.RunConfig(save_summary_steps=float('inf'),
                                  log_step_count_steps=args.quick_summary_period,
                                  keep_checkpoint_max=60)
                                  
  params = {
    'camera_size': args.camera_size,
    'small_camera_size': args.small_camera_size,
    'visual_size': args.visual_size,
    'model_dir': args.model_dir,
    'use_foveal': args.use_foveal,
    'random_fovea': args.random_fovea,
    'foveal_only': args.foveal_only,
    'attention_model_dir': args.attention_model_dir,
    'weight_data': args.weight_data,
    'epsilon': 1e-12,
    'learning_rate': args.learning_rate,
    'quick_summary_period': args.quick_summary_period,
    'slow_summary_period': args.slow_summary_period,
    'readout': args.readout,
    'augment_data': args.augment_data,
    'feature_map_size': args.feature_map_size,
    'gazemap_size': args.gazemap_size, 
    'stability_loss_weight': args.stability_loss_weight,
    'sample_fovea': args.sample_fovea,
    'attention_logit_factor': args.attention_logit_factor,
    'premade_attention_maps': args.premade_attention_maps,
    'premade_features': args.premade_features,
  }

  if args.visible_gpus is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpus
  
  model = tf.estimator.Estimator(
    model_fn=model_fn,
    model_dir=args.model_dir,
    config=config,
    params=params)
  
  # set up the directory to save the best checkpoint
  best_ckpt_dir = os.path.join(args.model_dir, 'best_ckpt')
  if not os.path.isdir(best_ckpt_dir) or len(os.listdir(best_ckpt_dir))==0:
    smallest_loss = float('Inf')
    if not os.path.isdir(best_ckpt_dir):
      os.makedirs(best_ckpt_dir)
  else:
    smallest_loss = [float(f.split('_')[1]) for f in os.listdir(best_ckpt_dir) if f.startswith('loss_')][0]
  
  for _ in range(args.train_epochs // args.epochs_before_validation):
    # Train the model.
    K.clear_session()
    model.train(input_fn=lambda: input_fn('training',
      args.batch_size, args.n_steps, 
      shuffle=True, include_labels=True, 
      n_epochs=args.epochs_before_validation, args=args,
      weight_data=args.weight_data,
      augment_data=args.augment_data)
    )
    # validate the model
    K.clear_session()
    valid_results = model.evaluate(input_fn=lambda: input_fn('validation', 
      batch_size=args.validation_batch_size, n_steps=args.validation_n_steps, 
      shuffle=False, include_labels=True, 
      n_epochs=1, args=args, weight_data=False) )
    print(valid_results)
    
    if valid_results['mae'] < smallest_loss:
      smallest_loss = valid_results['mae']
      # delete best_ckpt_dir
      shutil.rmtree(best_ckpt_dir)
      # re-make best_ckpt_dir as empty
      os.makedirs(best_ckpt_dir)
      # note down the new smallest loss
      open(os.path.join(best_ckpt_dir, 'loss_%f' % smallest_loss), 'a').close()
      # copy the checkpoint
      files_to_copy = [f for f in os.listdir(args.model_dir) 
        if f.startswith('model.ckpt-'+str(valid_results['global_step']))]
      for f in files_to_copy:
        shutil.copyfile(os.path.join(args.model_dir, f),
          os.path.join(best_ckpt_dir, f))
  
  


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  main(argv=sys.argv)
