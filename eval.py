import os
import torch
import argparse
import logging
from lib import evaluation


def main():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset', type=str, default='f30k', choices=['f30k', 'iapr_tc12'],
                        help='dataset name: f30k or iapr_tc12')
    parser.add_argument('--data_path', type=str, default='data/', help='the path of dataset')
    parser.add_argument('--save_results', type=int, default=0, help='whether save the results')
    parser.add_argument('--gpu-id', type=int, default=0, help='gpu id')
    parser.add_argument('--split', type=str, default=None, help='evaluation split, default follows dataset')
    parser.add_argument('--model_paths', nargs='*', default=None, help='explicit model paths to evaluate')

    opt = parser.parse_args()

    torch.cuda.set_device(opt.gpu_id)

    if opt.model_paths:
        weights_bases = opt.model_paths
    elif opt.dataset == 'f30k':
        weights_bases = [
            'runs/f30k_vsepp_shared',
            'runs/f30k_scan_shared',
            'runs/f30k_sgr_shared',
            'runs/f30k_chan_shared',
            'runs/f30k_laps',
        ]
    else:
        weights_bases = [
            'runs/iapr_tc12_vsepp_shared_vit',
            'runs/iapr_tc12_scan_shared_vit',
            'runs/iapr_tc12_sgr_shared_vit',
            'runs/iapr_tc12_chan_shared_vit',
            'runs/iapr_tc12_laps_vit',
        ]

    for base in weights_bases:

        # logging.basicConfig(filename=os.path.join(base, 'eval_extra.log'), filemode='w', 
        #                     format='%(asctime)s %(message)s', level=logging.DEBUG, force=True)
        logging.basicConfig()
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        
        logger.info('Evaluating {}...'.format(base))
        model_path = base if base.endswith('.pth') else os.path.join(base, 'model_best.pth')
        
        # Save the similarity matrix when requested.
        save_dir = os.path.dirname(model_path)
        save_path = os.path.join(save_dir, 'results_{}.npy'.format(opt.dataset)) if opt.save_results else None

        split = opt.split or 'test'
        evaluation.evalrank(model_path, data_path=opt.data_path, split=split, fold5=False, save_path=save_path)


if __name__ == '__main__':
    
    main()
