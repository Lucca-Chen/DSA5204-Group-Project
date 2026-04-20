import torch
import torch.nn as nn
import torch.nn.init
import lib.utils as utils
import logging
import arguments

from lib.encoders import get_image_encoder, get_text_encoder
from lib.loss import loss_select

from lib.cross_net import CrossSparseAggrNet_v2
from lib.sim_heads import build_similarity_head, chan_mean_similarity, global_similarity, scan_similarity

logger = logging.getLogger(__name__)


class VSEModel(nn.Module):
    def __init__(self, opt):
        super().__init__()

        self.opt = arguments.resolve_alignment_settings(opt)

        self.img_enc = get_image_encoder(self.opt)
        self.txt_enc = get_text_encoder(self.opt)

        self.criterion = loss_select(self.opt, loss_type=self.opt.loss)
        self.sim_head = self.opt.sim_head
        self.sim_head_module = build_similarity_head(self.opt)

        # iteration
        self.Eiters = 0

        # sparse + aggregation model for patch-word max-mean matching
        self.cross_net = CrossSparseAggrNet_v2(self.opt) if self.sim_head == 'max_mean' else None

    def freeze_backbone(self):
        self.img_enc.freeze_backbone()
        self.txt_enc.freeze_backbone()

    def unfreeze_backbone(self):
        self.img_enc.unfreeze_backbone()
        self.txt_enc.unfreeze_backbone()

    def set_max_violation(self, max_violation=True):
        if max_violation:
            self.criterion.max_violation_on()
        else:
            self.criterion.max_violation_off()

    # Compute the image and caption embeddings
    def forward_emb(self, images, captions, lengths):
       
        images = images.cuda()
        img_emb = self.img_enc(images)

        # compute caption embs
        captions = captions.cuda()
        lengths = lengths.cuda()
        
        cap_emb = self.txt_enc(captions, lengths)
        
        return img_emb, cap_emb, lengths
    
    # compute the similarity on cross-attention interaction
    def forward_sim(self, img_embs, cap_embs, cap_lens):

        if self.sim_head == 'max_mean':
            return self.cross_net(img_embs, cap_embs, cap_lens)

        if self.sim_head == 'global':
            sims = global_similarity(img_embs, cap_embs, cap_lens)
        elif self.sim_head in ('scan_t2i', 'scan_i2t', 'scan_all'):
            sims = scan_similarity(img_embs, cap_embs, cap_lens, scan_mode=self.sim_head)
        elif self.sim_head == 'chan_mean':
            sims = chan_mean_similarity(img_embs, cap_embs, cap_lens)
        elif self.sim_head_module is not None:
            sims = self.sim_head_module(img_embs, cap_embs, cap_lens)
        else:
            raise ValueError('Invalid sim_head {}'.format(self.sim_head))

        if self.training:
            return sims, None
        return sims

    # One training step given images and captions
    def forward(self, images, captions, lengths, img_ids=None, warmup_alpha=1.,):

        self.Eiters += 1
      
        img_emb = self.img_enc(images)
        cap_emb = self.txt_enc(captions, lengths)

        # get all samples for compute loss function
        # if self.opt.multi_gpu and (not self.opt.cross_attention):
        if self.opt.multi_gpu:
            lengths = utils.concat_all_gather(lengths, keep_grad=False)
            img_ids = utils.concat_all_gather(img_ids, keep_grad=False)
 
            max_len = int(lengths.max())
                
            if max_len > cap_emb.shape[1]:
                # (B, L_max - L, C)
                pad_emb = torch.zeros(cap_emb.shape[0], max_len - cap_emb.shape[1], cap_emb.shape[2], ).to(cap_emb.device)
                # (B, L, C) + (B, L_max - L, C) = (B, L_max, C)
                cap_emb = torch.cat([cap_emb, pad_emb], dim=1)

            # img_emb = utils.concat_all_gather(img_emb)
            # cap_emb = utils.concat_all_gather(cap_emb)
            img_emb = utils.all_gather_with_grad(img_emb)
            cap_emb = utils.all_gather_with_grad(cap_emb)            

        # compute similarity matrix
        improved_sims, score_mask_all = self.forward_sim(img_emb, cap_emb, lengths)

        # basic alignment loss
        align_loss = self.criterion(img_emb, cap_emb, img_ids, improved_sims) * warmup_alpha
        
        # ratio_loss is only meaningful when sparse token selection is enabled.
        if self.opt.use_ratio_loss and score_mask_all is not None:
            ratio_loss = (score_mask_all.mean() - self.opt.sparse_ratio) ** 2
        else:
            ratio_loss = torch.zeros([], device=align_loss.device)
        
        loss = align_loss + self.opt.ratio_weight * ratio_loss

        return loss


# optimizer init
def create_optimizer(opt, model):

    # Set up the lr for different parts of the VSE model
    decay_factor = 1e-4  
    cross_lr_rate = 1.0
        
    # bert params
    all_text_params = list(model.txt_enc.parameters())
    bert_params = list(model.txt_enc.bert.parameters())
    bert_params_ptr = [p.data_ptr() for p in bert_params]
    text_params_no_bert = list()

    for p in all_text_params:
        if p.data_ptr() not in bert_params_ptr:
            text_params_no_bert.append(p)

    # bert   
    params_list = [
        {'params': text_params_no_bert, 'lr': opt.learning_rate},
        {'params': bert_params, 'lr': opt.learning_rate * 0.1},
    ]

    # vit
    params_list += [
        {'params': model.img_enc.visual_encoder.parameters(), 'lr': opt.learning_rate * 0.1},
        {'params': model.img_enc.vision_proj.parameters(), 'lr': opt.learning_rate},
    ]

    # cross-moadl alignment 
    if model.cross_net is not None:
        params_list.append({'params': model.cross_net.parameters(), 'lr': opt.learning_rate * cross_lr_rate})
    if model.sim_head_module is not None:
        sim_head_params = list(model.sim_head_module.parameters())
        if sim_head_params:
            params_list.append({'params': sim_head_params, 'lr': opt.learning_rate * cross_lr_rate})

    params_list.append({'params': model.criterion.parameters(), 'lr': opt.learning_rate})
  
    optimizer = torch.optim.AdamW(params_list, lr=opt.learning_rate, weight_decay=decay_factor)
    
    return optimizer


if __name__ == '__main__':

    pass
