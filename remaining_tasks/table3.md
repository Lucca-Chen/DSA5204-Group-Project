# Table 3 Reproduction Instructions

This note records the complete command sequence for reproducing Table 3, the zero-shot visual grounding evaluation in the LAPS paper.

## Scope

The target table contains the following rows:

- Vanilla CLIP
- VSE++
- SCAN
- SGR
- LAPS

The target datasets are:

- RefCOCO: `val`, `testA`, `testB`
- RefCOCO+: `val`, `testA`, `testB`
- RefCOCOg: `val`, `test`

All commands below assume the current working directory is the repository root:

```bash
cd ./DSA5204-Group-Project
```

## 1. Download the required grounding data

This downloads:

- `COCO train2014`
- `RefCOCO`
- `RefCOCO+`
- `RefCOCOg`

```bash
qsub ./pbs/download_grounding_data_cpu.pbs
```

Useful checks:

```bash
qstat 1036271.stdct-mgmt-02
tail -f /scratch/e1553870/datasets/logs/grounding-data-1036271.stdct-mgmt-02.out
```

Expected locations after completion:

- COCO images: `/scratch/e1553870/datasets/coco-images`
- RefCOCO annotations: `/scratch/e1553870/datasets/refcoco_raw`

## 2. Train the CLIP-based retrieval models used in Table 3

Table 3 uses CLIP ViT-B/16 backbones trained on Flickr30K. The repository now provides one PBS template for all trained rows.

### VSE++

```bash
qsub -q batch_gpu -v MODEL_VARIANT=vsepp_shared ./pbs/train_f30k_clip_table3.pbs
```

### SCAN

```bash
qsub -q batch_gpu -v MODEL_VARIANT=scan_shared ./pbs/train_f30k_clip_table3.pbs
```

### SGR

```bash
qsub -q batch_gpu -v MODEL_VARIANT=sgr_shared ./pbs/train_f30k_clip_table3.pbs
```

### LAPS

```bash
qsub -q batch_gpu -v MODEL_VARIANT=laps ./pbs/train_f30k_clip_table3.pbs
```

The default output directory for these runs is:

```bash
/scratch/e1553870/DSA5204-Group-Project/runs_table3
```

The default PBS log directory is:

```bash
/scratch/e1553870/DSA5204-Group-Project/output/table3_train
```

To inspect the logs:

```bash
ls -lt /scratch/e1553870/DSA5204-Group-Project/output/table3_train | head
tail -f /scratch/e1553870/DSA5204-Group-Project/output/table3_train/<logfile>.out
```

## 3. Evaluate the Vanilla CLIP row

Vanilla CLIP does not require training.

```bash
qsub -q gpu -v MODEL_TYPE=vanilla_clip ./pbs/eval_table3_grounding.pbs
```

This writes the JSON result to:

```bash
/scratch/e1553870/DSA5204-Group-Project/grounding_results/vanilla_clip_all.json
```

## 4. Evaluate the trained rows

After each retrieval model finishes training, evaluate its grounding result with the corresponding checkpoint.

### VSE++

```bash
qsub -q gpu -v MODEL_TYPE=checkpoint,MODEL_VARIANT=vsepp_shared ./pbs/eval_table3_grounding.pbs
```

### SCAN

```bash
qsub -q gpu -v MODEL_TYPE=checkpoint,MODEL_VARIANT=scan_shared ./pbs/eval_table3_grounding.pbs
```

### SGR

```bash
qsub -q gpu -v MODEL_TYPE=checkpoint,MODEL_VARIANT=sgr_shared ./pbs/eval_table3_grounding.pbs
```

### LAPS

```bash
qsub -q gpu -v MODEL_TYPE=checkpoint,MODEL_VARIANT=laps ./pbs/eval_table3_grounding.pbs
```

The evaluator automatically looks for:

```bash
/scratch/e1553870/DSA5204-Group-Project/runs_table3/f30k_<variant>_clip_t3/model_best.pth
```

The default evaluation log directory is:

```bash
/scratch/e1553870/DSA5204-Group-Project/output/table3_eval
```

The default JSON output directory is:

```bash
/scratch/e1553870/DSA5204-Group-Project/grounding_results
```

Useful checks:

```bash
ls -lt /scratch/e1553870/DSA5204-Group-Project/output/table3_eval | head
tail -f /scratch/e1553870/DSA5204-Group-Project/output/table3_eval/<logfile>.out
```

## 5. Optional batch submission

To submit all training jobs at once:

```bash
./pbs/submit_table3_jobs.sh train_all
```

To submit all evaluation jobs at once:

```bash
./pbs/submit_table3_jobs.sh eval_all
```

## 6. Result files to collect

After all runs finish, collect these JSON files:

```bash
/scratch/e1553870/DSA5204-Group-Project/grounding_results/vanilla_clip_all.json
/scratch/e1553870/DSA5204-Group-Project/grounding_results/checkpoint_f30k_vsepp_shared_clip_t3.json
/scratch/e1553870/DSA5204-Group-Project/grounding_results/checkpoint_f30k_scan_shared_clip_t3.json
/scratch/e1553870/DSA5204-Group-Project/grounding_results/checkpoint_f30k_sgr_shared_clip_t3.json
/scratch/e1553870/DSA5204-Group-Project/grounding_results/checkpoint_f30k_laps_clip_t3.json
```

These files contain the per-dataset split scores needed to fill Table 3.
