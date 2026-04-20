from __future__ import print_function

import logging
import time
import torch
import numpy as np
import sys
from collections import OrderedDict

import arguments
from lib import utils
from lib import image_caption
from lib.tokenizers import get_tokenizer
from lib.vse import VSEModel


logger = logging.getLogger(__name__)


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        # current values
        self.val = val
        # total values
        self.sum += val * n
        # the number of records
        self.count += n
        # average values
        self.avg = self.sum / (.0001 + self.count)

    def __str__(self):

        # for values that should be recorded exactly e.g. iteration number
        if self.count == 0:
            return str(self.val)
        # for stats
        return '%.4f (%.4f)' % (self.val, self.avg)


class LogCollector(object):
    def __init__(self):
        # to keep the order of logged variables deterministic
        self.meters = OrderedDict()

    def update(self, k, v, n=0):
        # create a new meter if previously not recorded
        if k not in self.meters:
            self.meters[k] = AverageMeter()
        self.meters[k].update(v, n)

    def __str__(self):
        s = ''
        for i, (k, v) in enumerate(self.meters.items()):
            if i > 0:
                s += '  '
            s += k + ' ' + str(v)
        return s

    def tb_log(self, tb_logger, prefix='', step=None):
        for k, v in self.meters.items():
            tb_logger.log_value(prefix + k, v.val, step=step)


def encode_data(model, data_loader, log_step=10, logging=logger.info):

    batch_time = AverageMeter()
    val_logger = LogCollector()

    # switch to evaluate mode
    model.eval()

    end = time.time()

    # np array to keep all the embeddings
    img_embs = None
    cap_embs = None
    cap_img_ids = None
    
    # compute the number of max word
    max_n_word = model.opt.max_word

    for i, data_i in enumerate(data_loader):
        
        # make sure val logger is used       
        images, captions, lengths, ids, img_ids = data_i

        model.logger = val_logger

        # compute the embeddings
        with utils.get_autocast_context(model.opt):
            img_emb, cap_emb, lengths = model.forward_emb(images, captions, lengths)

        img_emb = img_emb.float()
        cap_emb = cap_emb.float()

        if img_embs is None:
            # for local visual features
            img_embs = torch.zeros((len(data_loader.dataset), img_emb.size(1), img_emb.size(2)))
            # for local textual features
            cap_embs = torch.zeros((len(data_loader.dataset), max_n_word, cap_emb.size(2)))
            
            cap_lens = torch.zeros(len(data_loader.dataset)).long()
            cap_img_ids = torch.zeros(len(data_loader.dataset)).long()

        # cache embeddings
        img_embs[ids] = img_emb.cpu()

        n_word = min(max(lengths), max_n_word)
        
        cap_embs[ids, :n_word, :] = cap_emb[:, :n_word, :].cpu()
        cap_lens[ids] = lengths.cpu()
        cap_img_ids[ids] = img_ids.cpu()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % log_step == 0:
            logging('Test: [{0}/{1}]\t'
                    '{e_log}\t'
                    'Batch-Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                .format(i, len(data_loader.dataset) // data_loader.batch_size + 1, batch_time=batch_time, e_log=str(model.logger)))
        del images, captions

    return img_embs, cap_embs, cap_lens, cap_img_ids


def build_caption_groups(cap_img_ids, n_images=None):

    if isinstance(cap_img_ids, torch.Tensor):
        cap_img_ids = cap_img_ids.cpu().numpy()
    cap_img_ids = np.asarray(cap_img_ids, dtype=np.int64)

    if n_images is None:
        n_images = int(cap_img_ids.max()) + 1 if cap_img_ids.size > 0 else 0

    groups = [np.where(cap_img_ids == image_index)[0] for image_index in range(n_images)]
    return groups


def select_unique_images(img_embs, cap_img_ids, n_images=None):

    if isinstance(cap_img_ids, torch.Tensor):
        cap_img_ids = cap_img_ids.cpu().numpy()
    cap_img_ids = np.asarray(cap_img_ids, dtype=np.int64)

    if n_images is None:
        n_images = int(cap_img_ids.max()) + 1 if cap_img_ids.size > 0 else 0

    keep = []
    for image_index in range(n_images):
        indices = np.where(cap_img_ids == image_index)[0]
        if len(indices) == 0:
            raise ValueError('Missing captions for image index {}'.format(image_index))
        keep.append(indices[0])

    keep = np.asarray(keep, dtype=np.int64)
    return img_embs[keep]


def evalrank(model_path, model=None, data_path=None, split='dev', fold5=False, save_path=None):

    # load model and options
    checkpoint = torch.load(model_path, map_location='cuda')
    opt = arguments.resolve_alignment_settings(checkpoint['opt'])

    logger.info(opt)

    # load vocabulary used by the model
    tokenizer = get_tokenizer(opt)

    # construct model
    if model is None:
        model = VSEModel(opt).cuda()

    # load model state
    model.load_state_dict(checkpoint['model'], strict=False)
    model.eval()

    logger.info('Loading dataset')
    data_loader = image_caption.get_test_loader(opt, data_path, tokenizer, 128, opt.workers, split)

    logger.info('Computing results...')
    with torch.no_grad():
        img_embs, cap_embs, cap_lens, cap_img_ids = encode_data(model, data_loader)

    n_images = int(cap_img_ids.max().item()) + 1 if len(cap_img_ids) > 0 else 0
    logger.info('Images: %d, Captions: %d' % (n_images, cap_embs.shape[0]))

    # for F30K, imgs 1000, captions 5000.
    # for COCO, imgs 5000, captions 25000. (5-fold is five times of 1000 imgs)

    if not fold5:
        caption_groups = build_caption_groups(cap_img_ids, n_images=n_images)
        img_embs = select_unique_images(img_embs, cap_img_ids, n_images=n_images)
        
        start = time.time()
        sims = shard_attn_scores(model, img_embs, cap_embs, cap_lens, opt).numpy()
        end = time.time()

        # npts = the number of images
        npts = img_embs.shape[0]

        if save_path is not None:
            np.save(save_path, {'npts': npts, 'sims': sims})
            logger.info('Save the similarity into {}'.format(save_path))

        logger.info("calculate similarity time: {}".format(end - start))

        r, rt = i2t(npts, sims, return_ranks=True, caption_groups=caption_groups)
        ri, rti = t2i(npts, sims, return_ranks=True, cap_img_ids=cap_img_ids.cpu().numpy())

        # r[0] -> R@1, r[1] -> R@5, r[2] -> R@10
        ar = (r[0] + r[1] + r[2]) / 3
        ari = (ri[0] + ri[1] + ri[2]) / 3

        rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
        logger.info("rsum: %.1f" % rsum)
        # logger.info("Average i2t Recall: %.1f" % ar)
        logger.info("Image to text (R@1, R@5, R@10): %.1f %.1f %.1f" % r[:3])
        # logger.info("Average t2i Recall: %.1f" % ari)
        logger.info("Text to image (R@1, R@5, R@10): %.1f %.1f %.1f" % ri[:3])
    
    else:
        # 5 fold cross-validation, only for MSCOCO
        results = []
        for i in range(5):
            img_embs_shard = img_embs[i * 5000:(i + 1) * 5000:5]
            cap_embs_shard = cap_embs[i * 5000:(i + 1) * 5000]
            cap_lens_shard = cap_lens[i * 5000:(i + 1) * 5000]
            
            start = time.time()
            sims = shard_attn_scores(model, img_embs_shard, cap_embs_shard, cap_lens_shard, opt).numpy()
            end = time.time()

            logger.info("calculate similarity time: {}".format(end - start))

            npts = img_embs_shard.shape[0]
            r, rt0 = i2t(npts, sims, return_ranks=True)
            ri, rti0 = t2i(npts, sims, return_ranks=True)

            logger.info("Image to text: %.1f, %.1f, %.1f" % r[:3])
            logger.info("Text to image: %.1f, %.1f, %.1f" % ri[:3])

            ar = (r[0] + r[1] + r[2]) / 3
            ari = (ri[0] + ri[1] + ri[2]) / 3
            rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
            # logger.info("rsum: %.1f ar: %.1f ari: %.1f" % (rsum, ar, ari))
            results += [list(r) + list(ri) + [ar, ari, rsum]]

        logger.info("-----------------------------------")
        logger.info("Mean metrics: ")
        mean_metrics = tuple(np.array(results).mean(axis=0).flatten())
        logger.info("rsum: %.1f" % (mean_metrics[12]))
        # logger.info("Average i2t Recall: %.1f" % mean_metrics[10])
        logger.info("Image to text (R@1, R@5, R@10): %.1f %.1f %.1f" % mean_metrics[:3])
        # logger.info("Average t2i Recall: %.1f" % mean_metrics[11])
        logger.info("Text to image (R@1, R@5, R@10): %.1f %.1f %.1f" % mean_metrics[5:8])


def i2t(npts, sims, return_ranks=False, mode='coco', caption_groups=None):

    ranks = np.zeros(npts)
    top1 = np.zeros(npts)

    if caption_groups is None:
        if mode == 'coco':
            caption_groups = [np.arange(5 * index, 5 * index + 5) for index in range(npts)]
        else:
            caption_groups = [np.asarray([index]) for index in range(npts)]

    for index in range(npts):
        
        inds = np.argsort(sims[index])[::-1]
        rank = 1e20
        for caption_index in caption_groups[index]:
            tmp = np.where(inds == caption_index)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank
        top1[index] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1

    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)


def t2i(npts, sims, return_ranks=False, mode='coco', cap_img_ids=None):

    if cap_img_ids is None:
        if mode == 'coco':
            cap_img_ids = np.repeat(np.arange(npts), 5)
        else:
            cap_img_ids = np.arange(npts)
    cap_img_ids = np.asarray(cap_img_ids, dtype=np.int64)

    ranks = np.zeros(len(cap_img_ids))
    top1 = np.zeros(len(cap_img_ids))

    # --> (5N(caption), N(image))
    sims = sims.T

    for caption_index in range(len(cap_img_ids)):
        inds = np.argsort(sims[caption_index])[::-1]
        image_index = cap_img_ids[caption_index]
        ranks[caption_index] = np.where(inds == image_index)[0][0]
        top1[caption_index] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)


def shard_attn_scores(model, img_embs, cap_embs, cap_lens, opt, gpu=False):

    shard_size = opt.shard_size
    n_im_shard = (len(img_embs) - 1) // shard_size + 1
    n_cap_shard = (len(cap_embs) - 1) // shard_size + 1

    sims = torch.zeros((len(img_embs), len(cap_embs)))
    if gpu:
        sims = sims.cuda()
    
    with torch.no_grad(): 
        
        for i in range(n_im_shard):    
            
            im_start, im_end = shard_size * i, min(shard_size * (i + 1), len(img_embs))

            for j in range(n_cap_shard):

                if utils.is_main_process():
                    sys.stdout.write('\r>> shard_attn_scores batch (%d,%d)' % (i, j))

                ca_start, ca_end = shard_size * j, min(shard_size * (j + 1), len(cap_embs))
                 
                im = img_embs[im_start:im_end].cuda()
                ca = cap_embs[ca_start:ca_end].cuda()
                l = cap_lens[ca_start:ca_end].long().cuda()

                with utils.get_autocast_context(model.opt):
                    sim = model.forward_sim(im, ca, l)
                if not gpu:
                    sim = sim.cpu()

                sims[im_start:im_end, ca_start:ca_end] = sim

    return sims


if __name__ == '__main__':

    pass
