import torch
import torch.utils.data as data
import os
import torchvision.transforms as T
import random
import json
import logging
import lib.utils as utils
from lib import tokenizers as tokenizer_utils
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)


DATASET_IMAGE_BASES = {
    'f30k': 'f30k_img_path',
    'coco': 'coco_img_path',
    'iapr_tc12': 'iapr_img_path',
    'rsicd': 'rsicd_img_path',
}


def build_transforms(img_size=224, is_train=True, is_clip=False):

    if is_clip:
        mean = [0.48145466, 0.4578275, 0.40821073]
        std = [0.26862954, 0.26130258, 0.27577711]
    else:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

    if not is_train:
        transform = T.Compose([
            T.Resize((img_size, img_size) , interpolation=Image.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
    else:
        transform = T.Compose([
            T.RandomResizedCrop(img_size, scale=(0.6, 1.0), interpolation=Image.BICUBIC),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    return transform


class RawImageDataset(data.Dataset):

    def __init__(self, opt, data_path, split, tokenizer, train):
        
        self.opt = opt

        self.train = train
        self.data_path = data_path
        self.split = split
        self.tokenizer = tokenizer
        self.train = train

        # f30k: 31014 imgs, 145000 train_captions
        # coco: 119287 imgs, 
        loc = os.path.join(opt.data_path, opt.dataset)

        image_base_key = DATASET_IMAGE_BASES.get(opt.dataset)
        if image_base_key is None:
            raise ValueError('Unsupported dataset {}'.format(opt.dataset))
        self.image_base = getattr(opt, image_base_key)

        with open(os.path.join(loc, 'id_mapping.json'), 'r') as f:
            self.id_to_path = json.load(f)

        # Read Captions
        self.captions = []
        # data_split: train or dev
        with open(os.path.join(loc, '%s_caps.txt' % self.split), 'r') as f:
            for line in f:
                self.captions.append(line.strip())

        # Get the image ids
        with open(os.path.join(loc, '{}_ids.txt'.format(self.split)), 'r') as f:
            image_ids = f.readlines()
            self.images = [int(x.strip()) for x in image_ids]

        capimgids_path = os.path.join(loc, '{}_capimgids.txt'.format(self.split))
        if os.path.isfile(capimgids_path):
            with open(capimgids_path, 'r') as f:
                self.caption_image_indices = [int(line.strip()) for line in f if line.strip()]
        else:
            if len(self.images) == 0:
                self.caption_image_indices = []
            elif len(self.images) == len(self.captions):
                self.caption_image_indices = list(range(len(self.captions)))
            elif len(self.captions) % len(self.images) == 0:
                captions_per_image = len(self.captions) // len(self.images)
                self.caption_image_indices = [index // captions_per_image for index in range(len(self.captions))]
            else:
                raise ValueError(
                    'Cannot infer caption-to-image mapping for {} {}. '
                    'Expected {}_capimgids.txt in {}'.format(opt.dataset, self.split, self.split, loc)
                )

        self.preprocess = build_transforms(
            img_size=opt.img_res,
            is_train=train,
            is_clip=('clip' in getattr(opt, 'vit_type', '')),
        )
        
        self.length = len(self.captions)
        self.num_images = len(self.images)
        if len(self.caption_image_indices) != self.length:
            raise ValueError(
                'Caption/image mapping length mismatch for {} {}: {} captions vs {} mappings'.format(
                    opt.dataset, self.split, self.length, len(self.caption_image_indices)
                )
            )
            
        print(opt.dataset, self.split)

    def __getitem__(self, index):
        
        img_index = self.caption_image_indices[index]
        caption = self.captions[index]
        target = tokenizer_utils.process_caption(self.tokenizer, caption, self.opt, train=self.train)

        image_id = self.images[img_index]
  
        image_path = os.path.join(self.image_base, self.id_to_path[str(image_id)])
        image = Image.open(image_path).convert("RGB")     
   
        image = self.preprocess(image)              

        return image, target, index, img_index

    def __len__(self):
        return self.length


def build_collate_fn(tokenizer):
    pad_token_id = tokenizer_utils.get_pad_token_id(tokenizer)

    def collate_fn_ours(data):
        # Sort a data list by caption length, for GRU/BERT/CLIP text encoder.
        data.sort(key=lambda x: len(x[1]), reverse=True)

        images, captions, ids, img_ids = zip(*data)

        img_ids = torch.tensor(img_ids)
        ids = torch.tensor(ids)
        images = torch.stack(images, 0)

        lengths = torch.tensor([len(cap) for cap in captions])
        targets = torch.full((len(captions), int(max(lengths))), fill_value=pad_token_id, dtype=torch.long)
        for i, cap in enumerate(captions):
            end = lengths[i]
            targets[i, :end] = cap[:end]

        return images, targets, lengths, ids, img_ids

    return collate_fn_ours


def get_loader(opt, data_path, split, tokenizer, 
               batch_size=128, shuffle=True, 
               num_workers=2, train=True,
               ):

    dataset = RawImageDataset(opt, data_path, split, tokenizer, train)
    collate_fn = build_collate_fn(tokenizer)

    # DDP with multi GPUS
    # only for train_loader
    if opt.multi_gpu and train:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()   
        sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank, shuffle=shuffle)
        shuffle = False
    else:
        sampler = None

    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                batch_size=batch_size,
                                                sampler=sampler,
                                                shuffle=shuffle,
                                                num_workers=num_workers,
                                                pin_memory=True,
                                                collate_fn=collate_fn,
                                                drop_last=train,
                                                )
    return data_loader


def get_train_loader(opt, data_path, tokenizer, batch_size, workers, split='train'):
    
    train_loader = get_loader(opt, data_path, split, tokenizer,
                              batch_size, True, workers, train=True)
    return train_loader


def get_test_loader(opt, data_path, tokenizer, batch_size, workers, split='test'):

    test_loader = get_loader(opt, data_path, split, tokenizer,
                             batch_size, False, workers, train=False) 
    return test_loader


if __name__ == '__main__':

    pass
