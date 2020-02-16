
import argparse
import sys
import os

import tensorflow as tf 

import networks

import add_args
from keras import backend as K
import shutil
import numpy as np
import pandas as pd
import feather
from tqdm import tqdm
from input_data import input_fn
from model import model_fn

import pdb


def main(argv):
  
  parser = argparse.ArgumentParser()
  add_args.for_general(parser)
  add_args.for_inference(parser)
  add_args.for_evaluation(parser)
  add_args.for_feature(parser)
  add_args.for_lstm(parser)
  args = parser.parse_args()
  
  config = tf.estimator.RunConfig(save_summary_steps=float('inf'))
                                  
  params = {
    'camera_size': args.camera_size,
    'small_camera_size': args.small_camera_size,
    'visual_size': args.visual_size,
    'model_dir': args.model_dir,
    'use_foveal': args.use_foveal,
    'foveal_only': args.foveal_only,
    'attention_model_dir': args.attention_model_dir,
    'weight_data': args.weight_data,
    'epsilon': 1e-12,
    'readout': args.readout,
    'output_details': True,
    'sample_fovea': args.sample_fovea,
    'attention_logit_factor': args.attention_logit_factor,
    'premade_attention_maps': args.premade_attention_maps,
    'premade_features': args.premade_features,
    'feature_map_size': args.feature_map_size,
    'gazemap_size': args.gazemap_size,
    'random_fovea': args.random_fovea,
  }

  if args.visible_gpus is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpus
  
  model = tf.estimator.Estimator(
    model_fn=model_fn,
    model_dir=args.model_dir,
    config=config,
    params=params)
  
  
  #determine which checkpoint to restore
  if args.model_iteration is None:
    best_ckpt_dir = os.path.join(args.model_dir, 'best_ckpt')
    if os.path.isdir(best_ckpt_dir):
      ckpt_name = [f.split('.index')[0] for f in os.listdir(best_ckpt_dir) if f.endswith('.index')][0]
      ckpt_path = os.path.join(best_ckpt_dir, ckpt_name)
      args.model_iteration = ckpt_name.split('-')[1]
  else:
    ckpt_name = 'model.ckpt-'+model_iteration
    ckpt_path = os.path.join(args.model_dir, ckpt_name)
  
  predict_generator = model.predict(
    input_fn = lambda: input_fn('test', 
      batch_size=args.batch_size, n_steps=args.n_steps, 
      shuffle=False, include_labels=True,
      n_epochs=1, args=args),
    checkpoint_path=ckpt_path)
  
  output_dir = os.path.join(args.model_dir, 'prediction_iter_'+args.model_iteration)
  if not os.path.isdir(output_dir):
    os.makedirs(output_dir)
  dfs = []
  video_ids = []
  for res in tqdm(predict_generator):
    n_steps = len(res['predicted_time_points'])
    video_id = res['video_id'].decode("utf-8")
    if '/' in video_id:
      video_id = video_id.split('/')[-1].split('.')[0]
    df = pd.DataFrame.from_dict({
            'video_key': [len(video_ids)]*n_steps,     # start from 0 but not 1
            'time_point': res['predicted_time_points'],
            'speed_x': res['speed'][:, 0],
            'speed_y': res['speed'][:, 1],
            'output_speed': res['output_speeds'][:, 0],
            'likelihood': res['likelihood'],
            'overlap': res['overlap'],
            })
    dfs.append(df)
    video_ids.append(video_id)
  
  output_df = pd.concat(dfs)
  feather.write_dataframe(output_df, os.path.join(output_dir, 'outputs.feather'))
  video_df = pd.DataFrame(
      data={
          'video_key': range(len(video_ids)),
          'video_id': video_ids,
      })
  feather.write_dataframe(video_df, os.path.join(output_dir, 'videos.feather'))


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  main(argv=sys.argv)
