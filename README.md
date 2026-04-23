# DSA5204 LAPS Reproduction and AITR Extension

This repository contains the DSA5204 project code for Linguistic-Aware Patch Slimming (LAPS) reproduction and the Adaptive Image--Text Retrieval (AITR) extension. The LAPS reproduction in the repository root is based on the official implementation of the CVPR 2024 paper "Linguistic-Aware Patch Slimming Framework for Fine-Grained Cross-Modal Alignment".

## Scope

The code supports:

- Flickr30K 1K retrieval with shared ViT-Base/224 + BERT-base and Swin-Base/224 + BERT-base backbones.
- The reproduced retrieval heads reported in the paper: VSE++, SCAN, SGR, CHAN, and LAPS.
- IAPR TC-12 evaluation under the shared ViT-Base/224 + BERT-base setting.
- AITR experiments with precomputed Faster R-CNN region features on Flickr30K and MS-COCO.

## Entry Points

The LAPS reproduction is run from the repository root:

```bash
python train.py --dataset f30k --data_path data/ --f30k_img_path /path/to/flickr30k-images --model_variant laps
python eval.py --dataset f30k --data_path data/ --model_paths runs/f30k_laps/model_best.pth
```

The AITR extension is run from `aitr/`:

```bash
cd aitr
python train.py --config configs/flickr30k_bert.yaml
python eval.py runs/flickr30k_bert/best.ckpt
```

## Environment

Recommended packages for the LAPS reproduction:

- Python 3.8+
- PyTorch 1.12+
- torchvision 0.13+
- transformers 4.32+
- opencv-python
- tensorboard_logger

The code loads pretrained BERT, ViT, and Swin models through Hugging Face:

- `bert-base-uncased`
- `google/vit-base-patch16-224-in21k`
- `microsoft/swin-base-patch4-window7-224`

AITR uses a separate dependency file:

```bash
cd aitr
python -m pip install -r requirements.txt
```

## Data Layout

### LAPS Data

Dataset files are not tracked in this repository. Place the normalized caption splits under `data/<dataset>/` and pass the image directory through the dataset-specific image path argument.

Flickr30K expects the following metadata files:

```text
data/
└── f30k/
    ├── train_ids.txt
    ├── train_caps.txt
    ├── dev_ids.txt
    ├── dev_caps.txt
    ├── test_ids.txt
    ├── test_caps.txt
    └── id_mapping.json
```

IAPR TC-12 can be converted into the same normalized format:

```bash
python scripts/prepare_iapr_tc12.py \
  --raw_root /path/to/iapr_tc12_raw \
  --image_root /path/to/iapr_tc12_images \
  --output_root data/iapr_tc12
```

### AITR Data

AITR follows the SCAN precomputed-feature layout with 36 Faster R-CNN regions per image:

```text
<DATA_ROOT>/
├── flickr30k/
│   └── precomp/
│       ├── train_ims.npy
│       ├── train_caps.txt
│       ├── dev_ims.npy
│       ├── dev_caps.txt
│       ├── test_ims.npy
│       └── test_caps.txt
└── coco/
    └── precomp/
        ├── train_ims.npy
        ├── train_caps.txt
        ├── dev_ims.npy
        ├── dev_caps.txt
        ├── test_ims.npy
        ├── test_caps.txt
        ├── testall_ims.npy
        └── testall_caps.txt
```

Set `data_root` in the corresponding AITR config file before training.

If a tabular annotation file is available:

```bash
python scripts/prepare_iapr_tc12.py \
  --raw_root /path/to/iapr_tc12_raw \
  --image_root /path/to/iapr_tc12_images \
  --annotation_file /path/to/annotations.tsv \
  --output_root data/iapr_tc12
```

## Flickr30K Reproduction

The shared-backbone comparison uses the same raw-image pipeline and BERT text encoder across retrieval heads.

```bash
python train.py --dataset f30k --data_path data/ --f30k_img_path /path/to/flickr30k-images --gpu-id 0 --logger_name runs/f30k_vsepp_shared --batch_size 64 --num_epochs 30 --vit_type vit --embed_size 512 --model_variant vsepp_shared
python train.py --dataset f30k --data_path data/ --f30k_img_path /path/to/flickr30k-images --gpu-id 0 --logger_name runs/f30k_scan_shared  --batch_size 64 --num_epochs 30 --vit_type vit --embed_size 512 --model_variant scan_shared
python train.py --dataset f30k --data_path data/ --f30k_img_path /path/to/flickr30k-images --gpu-id 0 --logger_name runs/f30k_sgr_shared   --batch_size 64 --num_epochs 30 --vit_type vit --embed_size 512 --model_variant sgr_shared
python train.py --dataset f30k --data_path data/ --f30k_img_path /path/to/flickr30k-images --gpu-id 0 --logger_name runs/f30k_chan_shared  --batch_size 64 --num_epochs 30 --vit_type vit --embed_size 512 --model_variant chan_shared
python train.py --dataset f30k --data_path data/ --f30k_img_path /path/to/flickr30k-images --gpu-id 0 --logger_name runs/f30k_laps         --batch_size 64 --num_epochs 30 --vit_type vit --embed_size 512 --model_variant laps --sparse_ratio 0.5 --aggr_ratio 0.4
```

For Swin-Base/224 + BERT-base, switch `--vit_type` to `swin`. The LAPS Swin setting uses `--sparse_ratio 0.8 --aggr_ratio 0.6`.

```bash
python train.py --dataset f30k --data_path data/ --f30k_img_path /path/to/flickr30k-images --gpu-id 0 --logger_name runs/f30k_laps_swin --batch_size 64 --num_epochs 30 --vit_type swin --embed_size 512 --model_variant laps --sparse_ratio 0.8 --aggr_ratio 0.6
```

Training writes `model_best.pth`, `checkpoint_last.pth`, `train.log`, `eval.log`, and `Parameters.txt` under the selected `--logger_name`.

## IAPR TC-12 Reproduction

After preparing `data/iapr_tc12`, train the same shared-backbone variants:

```bash
python train.py --dataset iapr_tc12 --data_path data/ --iapr_img_path /path/to/iapr_tc12_images --gpu-id 0 --logger_name runs/iapr_tc12_vsepp_shared_vit --batch_size 64 --num_epochs 30 --vit_type vit --embed_size 512 --model_variant vsepp_shared
python train.py --dataset iapr_tc12 --data_path data/ --iapr_img_path /path/to/iapr_tc12_images --gpu-id 0 --logger_name runs/iapr_tc12_scan_shared_vit  --batch_size 64 --num_epochs 30 --vit_type vit --embed_size 512 --model_variant scan_shared
python train.py --dataset iapr_tc12 --data_path data/ --iapr_img_path /path/to/iapr_tc12_images --gpu-id 0 --logger_name runs/iapr_tc12_sgr_shared_vit   --batch_size 64 --num_epochs 30 --vit_type vit --embed_size 512 --model_variant sgr_shared
python train.py --dataset iapr_tc12 --data_path data/ --iapr_img_path /path/to/iapr_tc12_images --gpu-id 0 --logger_name runs/iapr_tc12_chan_shared_vit  --batch_size 64 --num_epochs 30 --vit_type vit --embed_size 512 --model_variant chan_shared
python train.py --dataset iapr_tc12 --data_path data/ --iapr_img_path /path/to/iapr_tc12_images --gpu-id 0 --logger_name runs/iapr_tc12_laps_vit         --batch_size 64 --num_epochs 30 --vit_type vit --embed_size 512 --model_variant laps --sparse_ratio 0.5 --aggr_ratio 0.4
```

## AITR Extension

AITR is implemented as a separate subproject under `aitr/`. The main package is `aitr/aitr/`, and the training and evaluation entry points are `aitr/train.py` and `aitr/eval.py`.

The included configs cover the reported Faster R-CNN feature settings:

```bash
cd aitr
python train.py --config configs/flickr30k_bigru.yaml
python train.py --config configs/flickr30k_bert.yaml
python train.py --config configs/coco_bert.yaml
```

Each run writes its best checkpoint to the `ckpt_dir` specified in the config. Evaluate a checkpoint with:

```bash
python eval.py runs/flickr30k_bert/best.ckpt
python eval.py runs/coco_bert/best.ckpt
```

## Evaluation

LAPS training runs evaluation automatically at the end when `--eval 1`. Existing checkpoints can also be evaluated directly:

```bash
python eval.py --dataset f30k --data_path data/ --gpu-id 0 --model_paths runs/f30k_laps/model_best.pth
python eval.py --dataset iapr_tc12 --data_path data/ --gpu-id 0 --model_paths runs/iapr_tc12_laps_vit/model_best.pth
```

The evaluator reports I2T Recall@1/5/10, T2I Recall@1/5/10, and rSum.
