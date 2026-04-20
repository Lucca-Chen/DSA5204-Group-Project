import argparse
import json
import os

import torch

from lib.grounding import (
    VSEGroundingAdapter,
    VanillaCLIPGroundingAdapter,
    dump_grounding_results,
    evaluate_grounding_splits,
    load_vse_checkpoint,
)
from lib.refcoco import RefCOCODataset


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='checkpoint', choices=['checkpoint', 'vanilla_clip'])
    parser.add_argument('--checkpoint', type=str, default='', help='retrieval checkpoint for grounding evaluation')
    parser.add_argument('--clip_model_name', type=str, default='openai/clip-vit-base-patch16')
    parser.add_argument('--ref_root', type=str, default='/scratch/e1553870/datasets/refcoco_raw')
    parser.add_argument('--coco_img_path', type=str, default='/scratch/e1553870/datasets/coco-images')
    parser.add_argument('--dataset', type=str, default='all', choices=['all', 'refcoco', 'refcoco+', 'refcocog'])
    parser.add_argument('--grounding_split_by', type=str, default='')
    parser.add_argument('--grounding_iou_thresh', type=float, default=0.5)
    parser.add_argument('--grounding_alpha', type=float, default=1.0)
    parser.add_argument('--grounding_beta', type=float, default=0.0)
    parser.add_argument('--img_res', type=int, default=224)
    parser.add_argument('--vit_type', type=str, default='clip')
    parser.add_argument('--text_backbone', type=str, default='clip')
    parser.add_argument('--embed_size', type=int, default=512)
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--output_json', type=str, default='')
    return parser


def main():
    parser = build_parser()
    opt = parser.parse_args()

    device = torch.device(f'cuda:{opt.gpu_id}' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(opt.gpu_id)

    if opt.model_type == 'checkpoint':
        if not opt.checkpoint:
            raise ValueError('--checkpoint is required when --model_type checkpoint')
        model, ckpt_opt = load_vse_checkpoint(opt.checkpoint, device)
        ckpt_opt.ref_root = opt.ref_root
        ckpt_opt.coco_img_path = opt.coco_img_path
        ckpt_opt.grounding_split_by = opt.grounding_split_by
        adapter = VSEGroundingAdapter(model, ckpt_opt, device=device)
        data_opt = ckpt_opt
    else:
        adapter = VanillaCLIPGroundingAdapter(opt, device=device)
        data_opt = opt

    datasets = ['refcoco', 'refcoco+', 'refcocog'] if opt.dataset == 'all' else [opt.dataset]

    def dataset_builder(dataset_name, split):
        return RefCOCODataset(data_opt, dataset_name, split, split_by=opt.grounding_split_by)

    results = evaluate_grounding_splits(
        adapter,
        dataset_builder,
        datasets=datasets,
        alpha=opt.grounding_alpha,
        beta=opt.grounding_beta,
        iou_thresh=opt.grounding_iou_thresh,
    )

    print(json.dumps(results, indent=2))
    if opt.output_json:
        output_dir = os.path.dirname(opt.output_json)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        dump_grounding_results(results, opt.output_json)


if __name__ == '__main__':
    main()
