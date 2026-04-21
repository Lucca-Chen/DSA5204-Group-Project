# IAPR TC-12 and RSICD Extension Checklist

This note records the complete command sequence for running the lightweight extension experiments on `IAPR TC-12` and `RSICD`.

All commands below assume the current working directory is the repository root:

```bash
cd ./DSA5204-Group-Project
```

## 1. Prepare the raw datasets

### IAPR TC-12

The repository now includes a direct download job for the raw IAPR TC-12 archive.

```bash
qsub -q auto_free ./pbs/download_iapr_tc12_cpu.pbs
```

By default it downloads `mentalomega/iapr-tc12` from Kaggle and extracts:

```bash
/scratch/e1553870/datasets/iapr_tc12_raw
```

The preparation script then converts the raw benchmark into the repository retrieval format. It supports either:

- caption sidecar files such as `*.eng` or `*.txt`, or
- a structured annotation file passed through `ANNOTATION_FILE`

Run:

```bash
qsub ./pbs/prepare_iapr_tc12_cpu.pbs
```

Useful override when the image directory and annotation file are known explicitly:

```bash
qsub -v RAW_ROOT=/scratch/e1553870/datasets/iapr_tc12_raw,IMAGE_ROOT=/scratch/e1553870/datasets/iapr_tc12/images,ANNOTATION_FILE=/scratch/e1553870/datasets/iapr_tc12_raw/annotations.tsv ./pbs/prepare_iapr_tc12_cpu.pbs
```

The normalized dataset will be written to:

```bash
./data/iapr_tc12
```

### RSICD

Place the raw images and the standard `dataset_rsicd.json` file under:

```bash
/scratch/e1553870/datasets/rsicd
```

Run:

```bash
qsub ./pbs/prepare_rsicd_cpu.pbs
```

Useful override:

```bash
qsub -v ANNOTATION_JSON=/scratch/e1553870/datasets/rsicd_raw/dataset_rsicd.json,IMAGE_ROOT=/scratch/e1553870/datasets/rsicd/images ./pbs/prepare_rsicd_cpu.pbs
```

The normalized dataset will be written to:

```bash
./data/rsicd
```

## 2. Train retrieval models

The same PBS template is used for both datasets.

### IAPR TC-12

```bash
qsub -v DATASET=iapr_tc12,MODEL_VARIANT=vsepp_shared,VIT_TYPE=vit ./pbs/train_extra_dataset_shared_backbone.pbs
qsub -v DATASET=iapr_tc12,MODEL_VARIANT=scan_shared,VIT_TYPE=vit ./pbs/train_extra_dataset_shared_backbone.pbs
qsub -v DATASET=iapr_tc12,MODEL_VARIANT=sgr_shared,VIT_TYPE=vit ./pbs/train_extra_dataset_shared_backbone.pbs
qsub -v DATASET=iapr_tc12,MODEL_VARIANT=chan_shared,VIT_TYPE=vit ./pbs/train_extra_dataset_shared_backbone.pbs
qsub -v DATASET=iapr_tc12,MODEL_VARIANT=laps,VIT_TYPE=vit ./pbs/train_extra_dataset_shared_backbone.pbs
```

### RSICD

```bash
qsub -v DATASET=rsicd,MODEL_VARIANT=vsepp_shared,VIT_TYPE=vit ./pbs/train_extra_dataset_shared_backbone.pbs
qsub -v DATASET=rsicd,MODEL_VARIANT=scan_shared,VIT_TYPE=vit ./pbs/train_extra_dataset_shared_backbone.pbs
qsub -v DATASET=rsicd,MODEL_VARIANT=sgr_shared,VIT_TYPE=vit ./pbs/train_extra_dataset_shared_backbone.pbs
qsub -v DATASET=rsicd,MODEL_VARIANT=chan_shared,VIT_TYPE=vit ./pbs/train_extra_dataset_shared_backbone.pbs
qsub -v DATASET=rsicd,MODEL_VARIANT=laps,VIT_TYPE=vit ./pbs/train_extra_dataset_shared_backbone.pbs
```

### Swin variants

To switch from ViT to Swin, only change `VIT_TYPE`:

```bash
qsub -v DATASET=iapr_tc12,MODEL_VARIANT=laps,VIT_TYPE=swin ./pbs/train_extra_dataset_shared_backbone.pbs
qsub -v DATASET=rsicd,MODEL_VARIANT=laps,VIT_TYPE=swin ./pbs/train_extra_dataset_shared_backbone.pbs
```

To submit the full download → preparation → training → evaluation chain in one shot:

```bash
./pbs/submit_iapr_tc12_chain.sh
./pbs/submit_extra_dataset_jobs.sh train_all rsicd
./pbs/submit_extra_dataset_jobs.sh eval_all rsicd
```

Default outputs:

```bash
/scratch/e1553870/DSA5204-Group-Project/runs_extra
/scratch/e1553870/DSA5204-Group-Project/output/extra_train
```

## 3. Evaluate the trained checkpoints

### IAPR TC-12

```bash
qsub -v DATASET=iapr_tc12,MODEL_VARIANT=vsepp_shared,VIT_TYPE=vit ./pbs/eval_extra_dataset_shared_backbone.pbs
qsub -v DATASET=iapr_tc12,MODEL_VARIANT=scan_shared,VIT_TYPE=vit ./pbs/eval_extra_dataset_shared_backbone.pbs
qsub -v DATASET=iapr_tc12,MODEL_VARIANT=sgr_shared,VIT_TYPE=vit ./pbs/eval_extra_dataset_shared_backbone.pbs
qsub -v DATASET=iapr_tc12,MODEL_VARIANT=chan_shared,VIT_TYPE=vit ./pbs/eval_extra_dataset_shared_backbone.pbs
qsub -v DATASET=iapr_tc12,MODEL_VARIANT=laps,VIT_TYPE=vit ./pbs/eval_extra_dataset_shared_backbone.pbs
```

### RSICD

```bash
qsub -v DATASET=rsicd,MODEL_VARIANT=vsepp_shared,VIT_TYPE=vit ./pbs/eval_extra_dataset_shared_backbone.pbs
qsub -v DATASET=rsicd,MODEL_VARIANT=scan_shared,VIT_TYPE=vit ./pbs/eval_extra_dataset_shared_backbone.pbs
qsub -v DATASET=rsicd,MODEL_VARIANT=sgr_shared,VIT_TYPE=vit ./pbs/eval_extra_dataset_shared_backbone.pbs
qsub -v DATASET=rsicd,MODEL_VARIANT=chan_shared,VIT_TYPE=vit ./pbs/eval_extra_dataset_shared_backbone.pbs
qsub -v DATASET=rsicd,MODEL_VARIANT=laps,VIT_TYPE=vit ./pbs/eval_extra_dataset_shared_backbone.pbs
```

Default evaluation outputs:

```bash
/scratch/e1553870/DSA5204-Group-Project/output/extra_eval
```

Each evaluation writes retrieval scores back into the run directory through `eval.py`.

## 4. Expected normalized dataset structure

Both preparation scripts generate:

```text
data/<dataset>/
├── train_ids.txt
├── train_caps.txt
├── train_capimgids.txt
├── dev_ids.txt
├── dev_caps.txt
├── dev_capimgids.txt
├── test_ids.txt
├── test_caps.txt
├── test_capimgids.txt
└── id_mapping.json
```

The extra `*_capimgids.txt` file keeps the caption-to-image mapping explicit, which is needed for datasets that do not follow the fixed five-captions-per-image pattern of Flickr30K and MS-COCO.
