import argparse
import os


ALIGNMENT_PRESETS = {
    'laps': {
        'use_sparse': True,
        'use_aggr': True,
        'use_ratio_loss': True,
    },
    'basealign': {
        'use_sparse': False,
        'use_aggr': False,
        'use_ratio_loss': False,
    },
}

SIM_HEAD_PRESETS = {
    'global',
    'max_mean',
    'scan_all',
    'sgr',
    'chan_mean',
}

MODEL_VARIANTS = {
    'vsepp_shared': {
        'alignment_mode': 'basealign',
        'sim_head': 'global',
    },
    'scan_shared': {
        'alignment_mode': 'basealign',
        'sim_head': 'scan_all',
    },
    'sgr_shared': {
        'alignment_mode': 'basealign',
        'sim_head': 'sgr',
    },
    'chan_shared': {
        'alignment_mode': 'basealign',
        'sim_head': 'chan_mean',
    },
    'laps': {
        'alignment_mode': 'laps',
        'sim_head': 'max_mean',
    },
}

DATASET_SPLITS = {
    'f30k': {
        'val_split': 'dev',
        'test_split': 'test',
    },
    'iapr_tc12': {
        'val_split': 'dev',
        'test_split': 'test',
    },
}


def get_argument_parser():
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', default='data/', type=str, help='path to datasets')
    parser.add_argument('--dataset', default='f30k', choices=sorted(DATASET_SPLITS.keys()),
                        help='dataset name: f30k or iapr_tc12')

    parser.add_argument('--margin', default=0.2, type=float, help='Rank loss margin.')
    parser.add_argument('--num_epochs', default=30, type=int, help='Number of training epochs.')
    parser.add_argument('--batch_size', default=128, type=int, help='Size of a training mini-batch.')
    parser.add_argument('--embed_size', default=512, type=int, help='Dimensionality of the joint embedding.')
    
    parser.add_argument('--grad_clip', default=2., type=float, help='Gradient clipping threshold.')
    parser.add_argument('--learning_rate', default=2e-4, type=float, help='Initial learning rate.')
    
    parser.add_argument('--workers', default=8, type=int, help='Number of data loader workers.')
    parser.add_argument('--log_step', default=200, type=int, help='Number of steps to logger.info and record the log.')
    parser.add_argument('--val_step', default=500, type=int, help='Number of steps to run validation.')
    
    parser.add_argument('--logger_name', default='runs/test', help='Path to save Tensorboard log.')
    
    parser.add_argument('--max_violation', action='store_true', help='Use max instead of sum in the rank loss.')
    parser.add_argument('--vse_mean_warmup_epochs', type=int, default=1, help='The number of warmup epochs using mean vse loss')
    parser.add_argument('--embedding_warmup_epochs', type=int, default=0, help='The number of epochs for warming up the embedding layer')      
    
    parser.add_argument('--f30k_img_path', type=str, default='data/flickr30k-images', help='the path of f30k images') 
    parser.add_argument('--iapr_img_path', type=str, default='data/iapr_tc12/images',
                        help='the path of IAPR TC-12 images')
    
    # vision transformer
    parser.add_argument('--img_res', type=int, default=224, help='the image resolution for ViT input') 
    parser.add_argument('--vit_type', type=str, default='vit', help='the type of vit model')   

    # use DDP for training
    parser.add_argument('--multi_gpu', type=int, default=0, help='whether use multi-gpu for training')
    parser.add_argument('--world_size', type=int, default=1, help='number of distributed processes') 
    parser.add_argument("--rank", type=int, default=0, help='the parameter for rank')  
    parser.add_argument("--local_rank", type=int, default=0, help='the parameter for local rank')   

    parser.add_argument('--dist_backend', type=str, default='nccl', help='the backend for ddp')
    parser.add_argument('--dist_url', type=str, default='env://', help='url used to set up distributed training')
    parser.add_argument('--seed', type=int, default=0, help='fix the seed for reproducibility')

    # others
    parser.add_argument('--size_augment', type=int, default=1, help='whether use the Size Augmentation')
    parser.add_argument('--loss', type=str, default='vse', help='the objectve function for optimization')
    parser.add_argument('--eval', type=int, default=1, help='whether evaluation after training process')

    parser.add_argument('--save_results', type=int, default=1, help='whether save the evaluation results')
    parser.add_argument('--gpu-id', type=int, default=0, help='the gpu-id for runing')
    parser.add_argument('--resume', type=str, default='', help='checkpoint path to resume from')
    parser.add_argument('--save_last_checkpoint', type=int, default=1,
                        help='whether to save checkpoint_last.pth after each epoch')
    parser.add_argument('--amp', type=int, choices=[0, 1], default=0, help='whether to enable mixed precision')
    parser.add_argument('--amp_dtype', type=str, default='bf16', choices=['bf16', 'fp16'],
                        help='mixed precision dtype when --amp 1')

    parser.add_argument('--bert_path', type=str, default='bert-base-uncased')    
    parser.add_argument('--model_variant', type=str, default=None, choices=sorted(MODEL_VARIANTS.keys()),
                        help='shared-backbone preset, e.g. vsepp_shared, scan_shared, sgr_shared, chan_shared, or laps')
    parser.add_argument('--sim_head', type=str, default=None, choices=sorted(SIM_HEAD_PRESETS),
                        help='similarity head, default follows model_variant')
    parser.add_argument('--val_split', type=str, default=None, help='validation split, default follows dataset')
    parser.add_argument('--test_split', type=str, default=None, help='test split, default follows dataset')

    # optimizer
    parser.add_argument("--lr_schedules", default=[9, 15, 20, 25], type=int, nargs="+", help='epoch schedules for lr decay') 
    parser.add_argument("--decay_rate", default=0.3, type=float, help='lr decay_rate for optimizer') 

    parser.add_argument('--shard_size', type=int, default=256, help='the shard_size for cross-attention')   
    parser.add_argument('--max_word', type=int, default=90, help='the max length for word features')  
    parser.add_argument('--sim_dim', type=int, default=256, help='hidden dimensionality for learnable similarity heads')
    parser.add_argument('--sgr_step', type=int, default=3, help='number of graph reasoning steps for sgr head')
    parser.add_argument('--sgr_dropout', type=float, default=0.4, help='dropout for sgr-style global self-attention')

    # cross-modal alignment
    parser.add_argument('--alignment_mode', type=str, default=None, choices=sorted(ALIGNMENT_PRESETS.keys()),
                        help='preset for fair comparison: laps or basealign')
    parser.add_argument('--use_sparse', type=int, choices=[0, 1], default=None,
                        help='override sparse token selection, default follows alignment_mode')
    parser.add_argument('--use_aggr', type=int, choices=[0, 1], default=None,
                        help='override token aggregation, default follows alignment_mode')
    parser.add_argument('--use_ratio_loss', type=int, choices=[0, 1], default=None,
                        help='override ratio regularization, default follows alignment_mode')
    parser.add_argument('--aggr_ratio', type=float, default=0.4, help='the aggr rate for visual token')
    parser.add_argument('--sparse_ratio', type=float, default=0.5, help='the sparse rate for visual token') 
    parser.add_argument('--attention_weight', type=float, default=0.8, help='the weight of attention_map for mask prediction') 
    parser.add_argument('--ratio_weight', type=float, default=2.0, help='if use detach for kt loss')

    return parser


def resolve_alignment_settings(opt):

    variant_name = getattr(opt, 'model_variant', None)
    if variant_name is not None:
        variant = MODEL_VARIANTS[variant_name]
        if getattr(opt, 'alignment_mode', None) is None:
            opt.alignment_mode = variant['alignment_mode']
        if getattr(opt, 'sim_head', None) is None:
            opt.sim_head = variant['sim_head']
    elif getattr(opt, 'alignment_mode', None) is None:
        opt.alignment_mode = 'laps'

    if getattr(opt, 'sim_head', None) is None:
        opt.sim_head = 'max_mean'

    if opt.dataset in DATASET_SPLITS:
        split_defaults = DATASET_SPLITS[opt.dataset]
        if getattr(opt, 'val_split', None) is None:
            opt.val_split = split_defaults['val_split']
        if getattr(opt, 'test_split', None) is None:
            opt.test_split = split_defaults['test_split']

    if getattr(opt, 'model_variant', None) is None:
        for name, variant in MODEL_VARIANTS.items():
            if variant['alignment_mode'] == opt.alignment_mode and variant['sim_head'] == opt.sim_head:
                opt.model_variant = name
                break

    mode = getattr(opt, 'alignment_mode', 'laps')
    if mode not in ALIGNMENT_PRESETS:
        raise ValueError('Invalid alignment_mode {}'.format(mode))

    preset = ALIGNMENT_PRESETS[mode]

    for key, default in preset.items():
        value = getattr(opt, key, None)
        if value is None:
            value = default
        else:
            value = bool(value)
        setattr(opt, key, value)

    return opt


def save_parameters(opt, save_path):

    varx = vars(opt)
    base_str = ''
    for key in varx:
        base_str += str(key)
        if isinstance(varx[key], dict):
            for sub_key, sub_item in varx[key].items():
                base_str += '\n\t'+str(sub_key)+': '+str(sub_item)
        else:
            base_str += '\n\t'+str(varx[key])
        base_str+='\n'
    
    with open(os.path.join(save_path, 'Parameters.txt'), 'w') as f:
        f.write(base_str)


if __name__ == '__main__':

    pass
