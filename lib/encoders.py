import torch
import torch.nn as nn
from transformers import BertModel, CLIPTextModel, CLIPVisionModel, SwinModel, ViTModel
import logging

from lib.tokenizers import get_tokenizer


logger = logging.getLogger(__name__)


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.)


def get_image_encoder(opt):
    img_enc = VisionTransEncoder(opt)
    return img_enc


# ViT encoder
class VisionTransEncoder(nn.Module):
    def __init__(self, opt):
        super().__init__()

        self.opt = opt

        if 'clip' in opt.vit_type:
            self.visual_encoder = CLIPVisionModel.from_pretrained(opt.clip_model_name)
            image_size = int(self.visual_encoder.config.image_size)
            patch_size = int(self.visual_encoder.config.patch_size)
            opt.num_patches = (image_size // patch_size) ** 2
            print('clip vision model')
        elif 'swin' in opt.vit_type:
            # img_res 224 * 224, 7*7 patch
            self.visual_encoder = SwinModel.from_pretrained("microsoft/swin-base-patch4-window7-224")
            # self.visual_encoder = SwinModel.from_pretrained("../weights_models/microsoft--swin-base-patch4-window7-224")
            opt.num_patches = 49
            print('swin model')
        #  ViT model
        else:              
            # img_res 224 * 224, 14*14 patch
            self.visual_encoder = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
            # self.visual_encoder = ViTModel.from_pretrained("../weights_models/google--vit-base-patch16-224-in21k")
            opt.num_patches = 196
            print('vit model')

        # dimension transform
        if opt.embed_size == self.visual_encoder.config.hidden_size:
            self.vision_proj = nn.Identity()
        else:
            self.vision_proj = nn.Linear(self.visual_encoder.config.hidden_size, opt.embed_size)            

    def forward(self, images):
    
        # (B, L_v, C_hidden)
        img_feats = self.visual_encoder(images).last_hidden_state 

        # the dimension transform
        # (B, L_v, C)
        img_feats = self.vision_proj(img_feats)

        return img_feats  
        
    def freeze_backbone(self):
        for param in self.visual_encoder.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.visual_encoder.parameters():  
            param.requires_grad = True     


# Language Model with BERT backbone
class EncoderText_BERT(nn.Module):
    def __init__(self, opt):
        super(EncoderText_BERT, self).__init__()

        self.opt = opt
        self.embed_size = opt.embed_size
        
        self.tokenizer = get_tokenizer(opt)
        self.bert = BertModel.from_pretrained(opt.bert_path)
        
        if opt.embed_size == self.bert.config.hidden_size:
            self.fc = nn.Identity()
        else:
            self.fc = nn.Linear(self.bert.config.hidden_size, opt.embed_size)

    def forward(self, x, lengths):

        token_index = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        bert_attention_mask = (token_index < lengths.unsqueeze(1)).float()

        # all hidden features, D=768 in bert-base model
        # attention_mask： Mask to avoid performing attention on padding token indices.
        # bert_output[0] is the last/final hidden states of all tokens
        # bert_output[1] is the hidden state of [CLS] + one fc layer + Tanh, can be used for classification tasks.

        # N = max_cap_lengths, D = 768
        bert_emb = self.bert(input_ids=x, attention_mask=bert_attention_mask)[0]  # B x N x D

        # B x N x embed_size
        cap_emb = self.fc(bert_emb)
        
        return cap_emb        

    def freeze_backbone(self):
        for param in self.bert.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.bert.parameters():  
            param.requires_grad = True  


class EncoderText_CLIP(nn.Module):
    def __init__(self, opt):
        super().__init__()

        self.opt = opt
        self.embed_size = opt.embed_size
        self.tokenizer = get_tokenizer(opt)
        self.text_model = CLIPTextModel.from_pretrained(opt.clip_model_name)

        if opt.embed_size == self.text_model.config.hidden_size:
            self.fc = nn.Identity()
        else:
            self.fc = nn.Linear(self.text_model.config.hidden_size, opt.embed_size)

    def forward(self, x, lengths):
        token_index = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        attention_mask = token_index < lengths.unsqueeze(1)
        text_emb = self.text_model(input_ids=x, attention_mask=attention_mask)[0]
        return self.fc(text_emb)

    def freeze_backbone(self):
        for param in self.text_model.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.text_model.parameters():
            param.requires_grad = True


def get_text_encoder(opt):
    if getattr(opt, 'text_backbone', 'bert') == 'clip':
        return EncoderText_CLIP(opt)
    return EncoderText_BERT(opt)


if __name__ == '__main__':

    pass
