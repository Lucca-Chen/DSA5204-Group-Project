import json
import os
import pickle
from collections import defaultdict

import torch
import torch.utils.data as data
from PIL import Image

from lib.image_caption import build_transforms


DEFAULT_SPLIT_BY = {
    'refcoco': 'unc',
    'refcoco+': 'unc',
    'refcocog': 'umd',
}

DEFAULT_SPLITS = {
    'refcoco': ['val', 'testA', 'testB'],
    'refcoco+': ['val', 'testA', 'testB'],
    'refcocog': ['val', 'test'],
}


def _first_existing(paths):
    for path in paths:
        if path and os.path.exists(path):
            return path
    return None


def _load_refs(path):
    if path.endswith('.json'):
        with open(path, 'r') as f:
            return json.load(f)
    with open(path, 'rb') as f:
        try:
            return pickle.load(f, encoding='latin1')
        except TypeError:
            return pickle.load(f)


def _discover_annotation_files(ref_root, dataset, split_by=''):
    split_by = split_by or DEFAULT_SPLIT_BY.get(dataset, '')
    dataset_dir = os.path.join(ref_root, dataset)

    refs_path = _first_existing([
        os.path.join(dataset_dir, f'refs({split_by}).p'),
        os.path.join(dataset_dir, f'refs({split_by}).pkl'),
        os.path.join(dataset_dir, f'refs({split_by}).json'),
        os.path.join(dataset_dir, 'refs.p'),
        os.path.join(dataset_dir, 'refs.pkl'),
        os.path.join(dataset_dir, 'refs.json'),
    ])
    if refs_path is None:
        raise FileNotFoundError(f'Unable to find refs file for {dataset} under {dataset_dir}')

    instances_path = _first_existing([
        os.path.join(dataset_dir, 'instances.json'),
        os.path.join(ref_root, 'annotations', 'instances.json'),
        os.path.join(ref_root, dataset, 'annotations', 'instances.json'),
        os.path.join(ref_root, 'instances.json'),
    ])
    if instances_path is None:
        raise FileNotFoundError(f'Unable to find instances.json for {dataset} under {ref_root}')

    return refs_path, instances_path, split_by


def _resolve_image_path(coco_root, file_name):
    candidates = [
        os.path.join(coco_root, 'train2014', file_name),
        os.path.join(coco_root, 'val2014', file_name),
        os.path.join(coco_root, file_name),
    ]
    path = _first_existing(candidates)
    if path is None:
        raise FileNotFoundError(f'Unable to resolve COCO image path for {file_name} under {coco_root}')
    return path


def load_refcoco_samples(ref_root, dataset, coco_root, split=None, split_by=''):
    refs_path, instances_path, split_by = _discover_annotation_files(ref_root, dataset, split_by=split_by)
    refs = _load_refs(refs_path)
    with open(instances_path, 'r') as f:
        instances = json.load(f)

    anns = {ann['id']: ann for ann in instances['annotations']}
    images = {img['id']: img for img in instances['images']}

    proposals_by_image = defaultdict(list)
    for ann in instances['annotations']:
        proposals_by_image[ann['image_id']].append(ann['bbox'])

    samples = []
    for ref in refs:
        ref_split = ref.get('split', '')
        if split is not None and ref_split != split:
            continue

        ann_id = ref['ann_id']
        image_id = ref['image_id']
        image_info = images[image_id]
        gt_ann = anns[ann_id]
        image_path = _resolve_image_path(coco_root, image_info['file_name'])

        for sent in ref.get('sentences', []):
            expression = sent.get('sent') or sent.get('raw') or sent.get('sentence')
            if not expression:
                continue
            samples.append({
                'dataset': dataset,
                'split_by': split_by,
                'split': ref_split,
                'image_id': image_id,
                'image_path': image_path,
                'width': image_info['width'],
                'height': image_info['height'],
                'expression': expression,
                'gt_box': gt_ann['bbox'],
                'proposals': proposals_by_image[image_id],
            })

    return samples


class RefCOCODataset(data.Dataset):
    def __init__(self, opt, dataset, split, split_by=''):
        self.opt = opt
        self.dataset = dataset
        self.split = split
        self.split_by = split_by or getattr(opt, 'grounding_split_by', '') or DEFAULT_SPLIT_BY.get(dataset, '')
        self.samples = load_refcoco_samples(
            ref_root=opt.ref_root,
            dataset=dataset,
            coco_root=opt.coco_img_path,
            split=split,
            split_by=self.split_by,
        )
        self.transform = build_transforms(
            img_size=opt.img_res,
            is_train=False,
            is_clip=('clip' in getattr(opt, 'vit_type', '')),
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        image = Image.open(sample['image_path']).convert('RGB')
        image_tensor = self.transform(image)

        return {
            'image': image_tensor,
            'image_pil': image,
            'expression': sample['expression'],
            'gt_box': torch.tensor(sample['gt_box'], dtype=torch.float32),
            'proposals': torch.tensor(sample['proposals'], dtype=torch.float32),
            'image_id': sample['image_id'],
            'width': sample['width'],
            'height': sample['height'],
            'split': sample['split'],
            'dataset': sample['dataset'],
        }

