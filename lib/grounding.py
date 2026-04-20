import json
import math
from collections import defaultdict

import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPTokenizer

import arguments
from lib.refcoco import DEFAULT_SPLITS
from lib.sim_heads import get_spatial_tokens
from lib.tokenizers import get_tokenizer, process_caption
from lib.vse import VSEModel


def xywh_to_xyxy(boxes):
    boxes = boxes.clone()
    boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
    boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
    return boxes


def box_iou_xyxy(box_a, box_b):
    x1 = max(float(box_a[0]), float(box_b[0]))
    y1 = max(float(box_a[1]), float(box_b[1]))
    x2 = min(float(box_a[2]), float(box_b[2]))
    y2 = min(float(box_a[3]), float(box_b[3]))

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h

    area_a = max(0.0, float(box_a[2] - box_a[0])) * max(0.0, float(box_a[3] - box_a[1]))
    area_b = max(0.0, float(box_b[2] - box_b[0])) * max(0.0, float(box_b[3] - box_b[1]))
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def normalize_heatmap(heatmap):
    heatmap = heatmap - heatmap.min()
    denom = heatmap.max().clamp(min=1e-6)
    return heatmap / denom


def _box_to_grid(box_xywh, image_width, image_height, grid_width, grid_height):
    x, y, w, h = [float(v) for v in box_xywh]
    x1 = max(0, min(grid_width - 1, int(math.floor(x / image_width * grid_width))))
    y1 = max(0, min(grid_height - 1, int(math.floor(y / image_height * grid_height))))
    x2 = max(x1 + 1, min(grid_width, int(math.ceil((x + w) / image_width * grid_width))))
    y2 = max(y1 + 1, min(grid_height, int(math.ceil((y + h) / image_height * grid_height))))
    return x1, y1, x2, y2


def proposal_scores_from_heatmap(heatmap, boxes_xywh, image_width, image_height):
    grid_height, grid_width = heatmap.shape
    scores = []
    for box in boxes_xywh:
        x1, y1, x2, y2 = _box_to_grid(box, image_width, image_height, grid_width, grid_height)
        region = heatmap[y1:y2, x1:x2]
        scores.append(float(region.mean()))
    return torch.tensor(scores, dtype=torch.float32)


class VSEGroundingAdapter:
    def __init__(self, model, opt, device):
        self.model = model.eval()
        self.opt = opt
        self.device = device
        self.tokenizer = get_tokenizer(opt)

    def _tokenize(self, expression):
        target = process_caption(self.tokenizer, expression, self.opt, train=False)
        return target.unsqueeze(0).to(self.device), torch.tensor([len(target)], device=self.device)

    def _pair_score(self, img_emb, cap_emb, cap_lens):
        sims = self.model.forward_sim(img_emb, cap_emb, cap_lens)
        if isinstance(sims, tuple):
            sims = sims[0]
        return sims[0, 0]

    def heatmap_and_score(self, image_tensor, expression):
        token_ids, lengths = self._tokenize(expression)
        images = image_tensor.unsqueeze(0).to(self.device)

        img_emb, cap_emb, cap_lens = self.model.forward_emb(images, token_ids, lengths)
        img_emb.retain_grad()

        score = self._pair_score(img_emb, cap_emb, cap_lens)
        grads = torch.autograd.grad(score, img_emb, retain_graph=False, allow_unused=False)[0]

        token_feats = get_spatial_tokens(img_emb)
        token_grads = get_spatial_tokens(grads)
        heatmap = F.relu((token_feats * token_grads).sum(dim=-1))[0]

        grid_side = int(math.sqrt(heatmap.numel()))
        heatmap = heatmap.view(grid_side, grid_side).detach().cpu()
        heatmap = normalize_heatmap(heatmap)
        return heatmap, float(score.detach().cpu())

    def region_scores(self, image_pil, boxes_xywh, expression):
        scores = []
        token_ids, lengths = self._tokenize(expression)
        for box in boxes_xywh.tolist():
            x, y, w, h = [int(v) for v in box]
            crop = image_pil.crop((x, y, x + w, y + h))
            crop_tensor = self._preprocess_pil(crop).unsqueeze(0).to(self.device)
            with torch.enable_grad():
                img_emb, cap_emb, cap_lens = self.model.forward_emb(crop_tensor, token_ids, lengths)
                score = self._pair_score(img_emb, cap_emb, cap_lens)
            scores.append(float(score.detach().cpu()))
        return torch.tensor(scores, dtype=torch.float32)

    def _preprocess_pil(self, image_pil):
        from lib.image_caption import build_transforms

        transform = build_transforms(
            img_size=self.opt.img_res,
            is_train=False,
            is_clip=('clip' in getattr(self.opt, 'vit_type', '')),
        )
        return transform(image_pil.convert('RGB'))


class VanillaCLIPGroundingAdapter:
    def __init__(self, opt, device):
        self.opt = opt
        self.device = device
        self.model = CLIPModel.from_pretrained(opt.clip_model_name).to(device).eval()
        self.tokenizer = CLIPTokenizer.from_pretrained(opt.clip_model_name)

    def heatmap_and_score(self, image_tensor, expression):
        pixel_values = image_tensor.unsqueeze(0).to(self.device)
        pixel_values.requires_grad_(True)
        text_inputs = self.tokenizer(
            expression,
            truncation=True,
            max_length=getattr(self.tokenizer, 'model_max_length', 77),
            return_tensors='pt',
        )
        text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}

        outputs = self.model(pixel_values=pixel_values, **text_inputs, return_dict=True)
        score = outputs.logits_per_image[0, 0]
        grads = torch.autograd.grad(score, pixel_values, retain_graph=False, allow_unused=False)[0]

        heatmap = grads.abs().mean(dim=1, keepdim=True)
        grid_size = self.model.vision_model.config.image_size // self.model.vision_model.config.patch_size
        heatmap = F.interpolate(heatmap, size=(grid_size, grid_size), mode='bilinear', align_corners=False)[0, 0]
        heatmap = normalize_heatmap(heatmap.detach().cpu())
        return heatmap, float(score.detach().cpu())

    def region_scores(self, image_pil, boxes_xywh, expression):
        scores = []
        for box in boxes_xywh.tolist():
            x, y, w, h = [int(v) for v in box]
            crop = image_pil.crop((x, y, x + w, y + h))
            crop_tensor = self._preprocess_pil(crop).unsqueeze(0).to(self.device)
            text_inputs = self.tokenizer(
                expression,
                truncation=True,
                max_length=getattr(self.tokenizer, 'model_max_length', 77),
                return_tensors='pt',
            )
            text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
            with torch.no_grad():
                outputs = self.model(pixel_values=crop_tensor, **text_inputs, return_dict=True)
            scores.append(float(outputs.logits_per_image[0, 0].cpu()))
        return torch.tensor(scores, dtype=torch.float32)

    def _preprocess_pil(self, image_pil):
        from lib.image_caption import build_transforms

        transform = build_transforms(img_size=224, is_train=False, is_clip=True)
        return transform(image_pil.convert('RGB'))


def load_vse_checkpoint(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    opt = arguments.resolve_alignment_settings(checkpoint['opt'])
    model = VSEModel(opt).to(device)
    model.load_state_dict(checkpoint['model'], strict=False)
    model.eval()
    return model, opt


def select_best_box(adapter, sample, alpha=1.0, beta=0.0):
    heatmap, global_score = adapter.heatmap_and_score(sample['image'], sample['expression'])
    proposal_scores = proposal_scores_from_heatmap(
        heatmap,
        sample['proposals'],
        image_width=sample['width'],
        image_height=sample['height'],
    )

    if beta > 0:
        region_scores = adapter.region_scores(sample['image_pil'], sample['proposals'], sample['expression'])
        proposal_scores = alpha * proposal_scores + beta * region_scores

    best_idx = int(proposal_scores.argmax().item())
    return sample['proposals'][best_idx], heatmap, global_score


def evaluate_grounding_dataset(adapter, dataset, iou_thresh=0.5, alpha=1.0, beta=0.0):
    total = 0
    correct = 0
    for sample in dataset:
        pred_box_xywh, _, _ = select_best_box(adapter, sample, alpha=alpha, beta=beta)
        pred_box_xyxy = xywh_to_xyxy(pred_box_xywh.unsqueeze(0))[0]
        gt_box_xyxy = xywh_to_xyxy(sample['gt_box'].unsqueeze(0))[0]
        iou = box_iou_xyxy(pred_box_xyxy, gt_box_xyxy)
        correct += float(iou >= iou_thresh)
        total += 1
    return 100.0 * correct / max(total, 1)


def evaluate_grounding_splits(adapter, dataset_builder, datasets, alpha=1.0, beta=0.0, iou_thresh=0.5):
    results = defaultdict(dict)
    for dataset_name in datasets:
        for split in DEFAULT_SPLITS[dataset_name]:
            dataset = dataset_builder(dataset_name, split)
            score = evaluate_grounding_dataset(
                adapter,
                dataset,
                iou_thresh=iou_thresh,
                alpha=alpha,
                beta=beta,
            )
            results[dataset_name][split] = score
    return results


def dump_grounding_results(results, output_path):
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
