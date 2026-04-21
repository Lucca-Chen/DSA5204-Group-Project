#!/usr/bin/env python3
import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


def load_split_metadata(base_dir: Path):
    train_ids = base_dir.joinpath("train_ids.txt").read_text().splitlines()
    train_caps = base_dir.joinpath("train_caps.txt").read_text().splitlines()
    dev_ids = base_dir.joinpath("dev_ids.txt").read_text().splitlines()
    dev_caps = base_dir.joinpath("dev_caps.txt").read_text().splitlines()
    test_ids = base_dir.joinpath("test_ids.txt").read_text().splitlines()
    test_caps = base_dir.joinpath("test_caps.txt").read_text().splitlines()
    id_mapping = json.loads(base_dir.joinpath("id_mapping.json").read_text())

    sentences = defaultdict(list)
    splits = {}

    for idx, image_id in enumerate(train_ids):
        captions = train_caps[idx * 5:(idx + 1) * 5]
        sentences[image_id].extend(captions)
        splits[image_id] = "train"

    for image_id, caption in zip(dev_ids, dev_caps):
        sentences[image_id].append(caption)
        splits[image_id] = "val"

    for image_id, caption in zip(test_ids, test_caps):
        sentences[image_id].append(caption)
        splits[image_id] = "test"

    images = []
    for image_id_str, filename in sorted(id_mapping.items(), key=lambda item: int(item[0])):
        key = str(int(image_id_str))
        images.append({
            "filename": filename,
            "split": splits[key],
            "sentences": [{"raw": caption} for caption in sentences[key]],
        })

    return images, train_ids, dev_ids, test_ids


def export_feature_split(feature_array, box_array, split_ids, att_dir: Path, box_dir: Path):
    if len(feature_array) == len(split_ids):
        ordered_ids = split_ids
    else:
        ordered_ids = []
        seen = set()
        for image_id in split_ids:
            if image_id not in seen:
                ordered_ids.append(image_id)
                seen.add(image_id)

    for row_idx, image_id in enumerate(ordered_ids):
        feat_path = att_dir / f"{image_id}.npz"
        box_path = box_dir / f"{image_id}.npy"
        if not feat_path.exists():
            np.savez(feat_path, feat=np.asarray(feature_array[row_idx], dtype=np.float32))
        if not box_path.exists():
            np.save(box_path, np.asarray(box_array[row_idx], dtype=np.float32))


def build_sizes(images, images_root: Path, output_path: Path):
    sizes = []
    for item in images:
        with Image.open(images_root / item["filename"]) as image:
            sizes.append(image.size)
    with open(output_path, "wb") as handle:
        pickle.dump(sizes, handle)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_data_dir", default="/home/svu/e1553870/DSA5204-Group-Project/data/f30k")
    parser.add_argument("--feature_root", default="/scratch/e1553870/precomp/scan-features/data/data/f30k_precomp")
    parser.add_argument("--box_root", default="/scratch/e1553870/precomp/boxes/f30k")
    parser.add_argument("--images_root", default="/scratch/e1553870/datasets/flickr30k-images")
    parser.add_argument("--output_root", default="/scratch/e1553870/TGDT_data/f30k")
    args = parser.parse_args()

    base_dir = Path(args.base_data_dir)
    feature_root = Path(args.feature_root)
    box_root = Path(args.box_root)
    images_root = Path(args.images_root)
    output_root = Path(args.output_root)

    annotations_dir = output_root / "annotations"
    features_root = output_root / "features_36"
    att_dir = features_root / "bu_att"
    box_dir = features_root / "bu_box"
    images_link = output_root / "images"

    annotations_dir.mkdir(parents=True, exist_ok=True)
    att_dir.mkdir(parents=True, exist_ok=True)
    box_dir.mkdir(parents=True, exist_ok=True)

    if images_link.exists() or images_link.is_symlink():
        images_link.unlink()
    images_link.symlink_to(images_root, target_is_directory=True)

    images, train_ids, dev_ids, test_ids = load_split_metadata(base_dir)
    with open(annotations_dir / "dataset_flickr30k.json", "w") as handle:
        json.dump({"images": images}, handle)

    build_sizes(images, images_root, images_link / "sizes.pkl")

    export_feature_split(
        np.load(feature_root / "train_ims.npy", mmap_mode="r"),
        np.load(box_root / "train_boxes.npy", mmap_mode="r"),
        train_ids,
        att_dir,
        box_dir,
    )
    export_feature_split(
        np.load(feature_root / "dev_ims.npy", mmap_mode="r"),
        np.load(box_root / "dev_boxes.npy", mmap_mode="r"),
        dev_ids,
        att_dir,
        box_dir,
    )
    export_feature_split(
        np.load(feature_root / "test_ims.npy", mmap_mode="r"),
        np.load(box_root / "test_boxes.npy", mmap_mode="r"),
        test_ids,
        att_dir,
        box_dir,
    )

    print("Prepared TGDT Flickr30K data at", output_root)


if __name__ == "__main__":
    main()
