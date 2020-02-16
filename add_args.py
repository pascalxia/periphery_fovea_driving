def add_args(args, parser):
    for d in args:
        if 'nargs' not in d:
            d['nargs'] = None
        if 'required' not in d:
            d['required'] = False
        parser.add_argument('--'+d['name'],
                            nargs=d['nargs'],
                            default=d['default'],
                            type=d['type'],
                            help=d['help'],
                            required=d['required'])
                            
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def for_general(parser):
    args = [
    {
     'name': 'data_dir',
     'default': 'data',
     'type': str,
     'help': 'folder of dataset'},
    {
     'name': 'model_dir',
     'default': None,
     'type': str,
     'help': 'folder from which restore the model '},
    {
     'name': 'use_foveal',
     'default': False,
     'type': str2bool,
     'help': 'Whether to use foveal vision'},
    {
     'name': 'random_fovea',
     'default': False,
     'type': str2bool,
     'help': 'Whether uniformly randomly choose foveae'},
    {
     'name': 'sample_fovea',
     'default': False,
     'type': str2bool,
     'help': 'Whether use sampling to decide where the foveae are'},
    {
     'name': 'attention_logit_factor',
     'default': None,
     'type': float,
     'help': 'the facotr that is multiplied to the attention logits for fovea sampling'},
    {
     'name': 'foveal_only',
     'default': False,
     'type': str2bool,
     'help': 'Whether to only use foveal vision'},
    {
     'name': 'attention_model_dir',
     'default': None,
     'type': str,
     'help': 'folder from which restore the attention model '},
    {
     'name': 'premade_attention_maps',
     'default': False,
     'type': str2bool,
     'help': 'Whether the input tfrecords contain premade attention maps'},
    {
     'name': 'premade_features',
     'default': False,
     'type': str2bool,
     'help': 'Whether the input tfrecords contain premade features'},
    {
     'name': 'camera_size',
     'nargs': 2,
     'default': [576,1024],
     'type': int,
     'help': 'Size of the input image'},
    {
     'name': 'small_camera_size',
     'nargs': 2,
     'default': [72,128],
     'type': int,
     'help': 'Size of the input image for perpheral vision'},
    {
     'name': 'visual_size',
     'nargs': 2,
     'default': [288,512],
     'type': int,
     'help': 'Size of the images visualized in Tensorboard'},
    {
     'name': 'gazemap_size',
     'nargs': 2,
     'default': [36,64],
     'type': int,
     'help': 'Size of the predicted gaze map'},
    {
     'name': 'weight_data',
     'default': False,
     'type': str2bool,
     'help': 'whether to weight the data points differently in trianing'},
    {
     'name': 'visible_gpus',
     'default': None,
     'type': str,
     'help': 'GPUs that are visible to Tensorflow, e.g., 0,1'},
    {
     'name': 'num_parallel',
     'default': 10,
     'type': int,
     'help': 'number of parellel threads'},
    {
     'name': 'pad_batch',
     'default': True,
     'type': str2bool,
     'help': 'whether to pad each batch'},
    {
     'name': 'check_long_enough',
     'default': True,
     'type': str2bool,
     'help': 'whether to check the sequence in tfrecords is long enough to contain the future to predict'},
    {
     'name': 'discrete_output',
     'default': False,
     'type': str2bool,
     'help': 'whether to predict discrete action categories'},
    {
     'name': 'multiple_tfrecords',
     'default': False,
     'type': str2bool,
     'help': 'whether a long video is divided into multiple tfrecords'},
    ]
    add_args(args, parser)


def for_inference(parser):
    args = [
    {
     'name': 'batch_size',
     'default': 20,
     'type': int,
     'help': 'basic batch size'},
    {
     'name': 'validation_batch_size',
     'default': 1,
     'type': int,
     'help': 'batch size used during validation'},
    {
     'name': 'use_prior',
     'default': False,
     'type': str2bool,
     'help': 'whether to use prior gaze map'},
    {
     'name': 'drop_rate',
     'default': 0,
     'type': float,
     'help': 'drop rate'},
    {
     'name': 'readout',
     'default': 'default',
     'type': str,
     'help': 'which readout network to use'},
    {
     'name': 'sparsity_weight',
     'default': 0,
     'type': float,
     'help': 'The weight of sparsity regularization'}, 
    {
     'name': 'gpu_memory_fraction',
     'default': None,
     'type': float,
     'help': 'The fraction of GPU memory to use'},
     {
     'name': 'binary',
     'default': False,
     'type': str2bool,
     'help': 'Whether to make the gaze maps to binary maps'},
     {
     'name': 'annotation_threshold',
     'default': None,
     'type': float,
     'help': 'When the gaze density is more than annotation_threshold times the uniform density, the pixel is gazed'}
    ]
    add_args(args, parser)
    
    
def for_feature(parser):
    args = [
    {
     'name': 'feature_name',
     'default': 'alexnet',
     'type': str,
     'help': 'Which kind of features to use'},
    {
     'name': 'feature_map_size',
     'nargs': 2,
     'default': [9, 16],
     'type': int,
     'help': 'Feature map size (not include the number of channels)'},
    {
     'name': 'input_feature_map_size',
     'nargs': 2,
     'default': [3, 7],
     'type': int,
     'help': 'Input feature map (not interpolated) size (not include the number of channels)'},
    {
     'name': 'feature_map_channels',
     'default': 2560,
     'type': int,
     'help': 'The number of feature map channels'}
    ]
    add_args(args, parser)
    
    
def for_training(parser):
    args = [
    {
     'name': 'learning_rate',
     'default': 1e-3,
     'type': float,
     'help': 'Learning rate for Adam Optimizer'},
    {
     'name': 'max_iteration',
     'default': 10001,
     'type': int,
     'help': 'Maximum iterations'},
    {
     'name': 'train_epochs',
     'default': 10,
     'type': int,
     'help': 'For how many epochs the model should be trained in total'},
    {
     'name': 'epochs_before_validation',
     'default': 1,
     'type': int,
     'help': 'For how many epochs the model should be trained before each time of validation'},
    {
     'name': 'quick_summary_period',
     'default': 10,
     'type': int,
     'help': 'After how many iterations do some quick summaries'},
    {
     'name': 'slow_summary_period',
     'default': 50,
     'type': int,
     'help': 'After how many iterations do some slow summaries'},
    {
     'name': 'valid_summary_period',
     'default': 500,
     'type': int,
     'help': 'After how many iterations do validation and save one checkpoint'},
    {
     'name': 'valid_batch_factor',
     'default': 2,
     'type': int,
     'help': 'The batch size for validation is equal to this number multiply the original batch size'},
    {
     'name': 'logs_dir',
     'default': None,
     'type': str,
     'help': 'path to logs directory'},
    {
     'name': 'augment_data',
     'default': False,
     'type': str2bool,
     'help': 'whether to use data augmentation during training'},
    {
     'name': 'stability_loss_weight',
     'default': 0,
     'type': float,
     'help': 'The weight for the stability loss (error derivative loss)'},
    ]
    add_args(args, parser)
    
    
def for_evaluation(parser):
    args = [
    {
     'name': 'model_iteration',
     'default': None,
     'type': str,
     'help': 'The model of which iteration to resotre'},
    ]
    add_args(args, parser)


def for_lstm(parser):
    args = [
    {
     'name': 'n_steps',
     'default': None,
     'type': int,
     'help': 'number of time steps for each sequence'},
    {
     'name': 'validation_n_steps',
     'default': None,
     'type': int,
     'help': 'number of time steps for each sequence during validation'},
    {
     'name': 'longest_seq',
     'default': None,
     'type': int,
     'help': 'How many frames can the longest sequence contain'},
    {
      'name': 'n_future_steps',
      'default': 0,
      'type': int,
      'help': 'predict how many steps in the future'}
    ]
    add_args(args, parser)
