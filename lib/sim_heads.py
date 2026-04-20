import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.xttn import xattn_score_i2t, xattn_score_t2i, xattn_score_two


def _is_square(num_tokens):
    side = int(math.sqrt(num_tokens))
    return side * side == num_tokens


def has_cls_token(img_embs):
    token_count = img_embs.size(1)
    return token_count > 1 and (not _is_square(token_count)) and _is_square(token_count - 1)


def get_spatial_tokens(img_embs):
    if has_cls_token(img_embs):
        return img_embs[:, 1:, :]
    return img_embs


def get_text_mask(cap_embs, cap_lens):
    max_len = cap_embs.size(1)
    cap_lens = cap_lens.clamp(max=max_len)
    token_index = torch.arange(max_len, device=cap_embs.device).unsqueeze(0)
    return token_index < cap_lens.unsqueeze(1)


def l1norm(x, dim, eps=1e-8):
    return x / (torch.norm(x, p=1, dim=dim, keepdim=True) + eps)


def scan_attention(query, context, smooth=9.0):
    query_t = torch.transpose(query, 1, 2)
    attn = torch.bmm(context, query_t)
    attn = F.leaky_relu(attn, negative_slope=0.1)
    attn = F.normalize(attn, dim=2)
    attn = torch.transpose(attn, 1, 2).contiguous()
    attn = F.softmax(attn * smooth, dim=2)
    attn_t = torch.transpose(attn, 1, 2).contiguous()
    context_t = torch.transpose(context, 1, 2)
    weighted_context = torch.bmm(context_t, attn_t)
    weighted_context = torch.transpose(weighted_context, 1, 2)
    weighted_context = F.normalize(weighted_context, dim=-1)
    return weighted_context


def global_similarity(img_embs, cap_embs, cap_lens):
    img_tokens = get_spatial_tokens(img_embs)
    img_global = F.normalize(img_tokens.mean(dim=1), dim=-1)

    cap_mask = get_text_mask(cap_embs, cap_lens).unsqueeze(-1).to(cap_embs.dtype)
    cap_global = (cap_embs * cap_mask).sum(dim=1) / cap_mask.sum(dim=1).clamp(min=1.0)
    cap_global = F.normalize(cap_global, dim=-1)

    return img_global.mm(cap_global.t())


def scan_similarity(img_embs, cap_embs, cap_lens, scan_mode='scan_all'):
    img_tokens = get_spatial_tokens(img_embs)

    if scan_mode == 'scan_t2i':
        return xattn_score_t2i(img_tokens, cap_embs, cap_lens)
    if scan_mode == 'scan_i2t':
        return xattn_score_i2t(img_tokens, cap_embs, cap_lens)
    if scan_mode == 'scan_all':
        return xattn_score_two(img_tokens, cap_embs, cap_lens)

    raise ValueError('Invalid scan_mode {}'.format(scan_mode))


def chan_mean_similarity(img_embs, cap_embs, cap_lens):
    img_tokens = F.normalize(get_spatial_tokens(img_embs), dim=-1)
    cap_tokens = F.normalize(cap_embs, dim=-1)

    similarities = []
    n_image = img_tokens.size(0)
    n_caption = cap_tokens.size(0)

    for i in range(n_caption):
        n_word = int(cap_lens[i].item())
        cap_i = cap_tokens[i, :n_word, :].unsqueeze(0).repeat(n_image, 1, 1)
        # (n_image, n_region, n_word)
        token_sims = torch.bmm(img_tokens, cap_i.transpose(1, 2))
        # Word-to-region hard assignment, then mean pool over valid words.
        hard_assign = token_sims.max(dim=1)[0]
        sim_i = hard_assign.mean(dim=1, keepdim=True)
        similarities.append(sim_i)

    return torch.cat(similarities, dim=1)


class GlobalSelfAttention(nn.Module):
    def __init__(self, embed_dim, dropout=0.4):
        super().__init__()
        self.local_proj = nn.Linear(embed_dim, embed_dim)
        self.global_proj = nn.Linear(embed_dim, embed_dim)
        self.common_proj = nn.Linear(embed_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        for module in (self.local_proj, self.global_proj, self.common_proj):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, local, raw_global, mask=None):
        local_emb = torch.tanh(self.local_proj(local))
        global_emb = torch.tanh(self.global_proj(raw_global))
        local_emb = self.dropout(local_emb)
        global_emb = self.dropout(global_emb)

        logits = self.common_proj(local_emb * global_emb.unsqueeze(1)).squeeze(-1)
        if mask is not None:
            logits = logits.masked_fill(~mask, -1e4)
        weights = torch.softmax(logits, dim=1)
        global_feat = (weights.unsqueeze(-1) * local).sum(dim=1)
        return F.normalize(global_feat, dim=-1)


class GraphReasoning(nn.Module):
    def __init__(self, sim_dim):
        super().__init__()
        self.graph_query_w = nn.Linear(sim_dim, sim_dim)
        self.graph_key_w = nn.Linear(sim_dim, sim_dim)
        self.sim_graph_w = nn.Linear(sim_dim, sim_dim)
        self.relu = nn.ReLU()
        self.reset_parameters()

    def reset_parameters(self):
        for module in (self.graph_query_w, self.graph_key_w, self.sim_graph_w):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, sim_emb):
        sim_query = self.graph_query_w(sim_emb)
        sim_key = self.graph_key_w(sim_emb)
        sim_edge = torch.softmax(torch.bmm(sim_query, sim_key.transpose(1, 2)), dim=-1)
        sim_sgr = torch.bmm(sim_edge, sim_emb)
        return self.relu(self.sim_graph_w(sim_sgr))


class SGRSimilarityHead(nn.Module):
    def __init__(self, embed_size, sim_dim=256, sgr_step=3, dropout=0.4):
        super().__init__()
        self.v_global_w = GlobalSelfAttention(embed_size, dropout=dropout)
        self.t_global_w = GlobalSelfAttention(embed_size, dropout=dropout)
        self.sim_tranloc_w = nn.Linear(embed_size, sim_dim)
        self.sim_tranglo_w = nn.Linear(embed_size, sim_dim)
        self.sim_eval_w = nn.Linear(sim_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.sgr_layers = nn.ModuleList([GraphReasoning(sim_dim) for _ in range(sgr_step)])
        self.reset_parameters()

    def reset_parameters(self):
        for module in (self.sim_tranloc_w, self.sim_tranglo_w, self.sim_eval_w):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, img_embs, cap_embs, cap_lens):
        img_tokens = get_spatial_tokens(img_embs)
        img_avg = img_tokens.mean(dim=1)
        img_global = self.v_global_w(img_tokens, img_avg)

        similarities = []
        n_image = img_tokens.size(0)
        n_caption = cap_embs.size(0)

        for i in range(n_caption):
            n_word = int(cap_lens[i].item())
            cap_i = cap_embs[i, :n_word, :].unsqueeze(0)
            cap_i_expand = cap_i.repeat(n_image, 1, 1)

            cap_avg_i = cap_i.mean(dim=1).repeat(n_image, 1)
            cap_global_i = self.t_global_w(cap_i_expand, cap_avg_i)

            context_img = scan_attention(cap_i_expand, img_tokens, smooth=9.0)
            sim_loc = torch.pow(context_img - cap_i_expand, 2)
            sim_loc = F.normalize(self.sim_tranloc_w(sim_loc), dim=-1)

            sim_glo = torch.pow(img_global - cap_global_i, 2)
            sim_glo = F.normalize(self.sim_tranglo_w(sim_glo), dim=-1)

            sim_emb = torch.cat([sim_glo.unsqueeze(1), sim_loc], dim=1)
            for layer in self.sgr_layers:
                sim_emb = layer(sim_emb)
            sim_vec = sim_emb[:, 0, :]
            sim_i = self.sigmoid(self.sim_eval_w(sim_vec))
            similarities.append(sim_i)

        return torch.cat(similarities, dim=1)


def build_similarity_head(opt):
    if opt.sim_head == 'sgr':
        return SGRSimilarityHead(
            embed_size=opt.embed_size,
            sim_dim=opt.sim_dim,
            sgr_step=opt.sgr_step,
            dropout=opt.sgr_dropout,
        )
    return None
