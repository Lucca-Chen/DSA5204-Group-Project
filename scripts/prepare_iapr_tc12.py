#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from dataset_prep_common import (
    build_image_index,
    find_image_relpath,
    normalize_split_name,
    read_table_annotations,
    split_records,
    write_retrieval_dataset,
)


CAPTION_KEYS = ('caption', 'captions', 'description', 'descriptions', 'text', 'sentence', 'eng', 'english')
IMAGE_KEYS = ('image', 'image_name', 'image_id', 'filename', 'file_name', 'file', 'img')


def guess_image_root(raw_root):
    raw_root = Path(raw_root)
    candidates = [
        raw_root / 'images',
        raw_root / 'image',
        raw_root / 'imgs',
        raw_root / 'img',
        raw_root,
    ]
    for candidate in candidates:
        if candidate.is_dir():
            for path in candidate.rglob('*'):
                if path.is_file() and path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}:
                    return candidate
    raise FileNotFoundError(f'Could not locate an image directory under {raw_root}')


def iter_caption_texts(raw_root):
    raw_root = Path(raw_root)
    for suffix in ('.eng', '.txt'):
        for path in sorted(raw_root.rglob(f'*{suffix}')):
            if path.is_file():
                yield path


def extract_captions_from_row(row):
    captions = []
    for key in CAPTION_KEYS:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            captions.append(value)
        elif isinstance(value, list):
            captions.extend(str(item) for item in value if str(item).strip())
    return captions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_root', required=True, help='root directory of the unpacked IAPR TC-12 data')
    parser.add_argument('--image_root', default='', help='optional explicit image root')
    parser.add_argument('--annotation_file', default='', help='optional structured annotation file (json/jsonl/csv/tsv/txt)')
    parser.add_argument('--output_root', default='data/iapr_tc12', help='output directory in repository format')
    parser.add_argument('--seed', type=int, default=0, help='seed used for the deterministic split')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='training split ratio when no official split is provided')
    parser.add_argument('--dev_ratio', type=float, default=0.1, help='validation split ratio when no official split is provided')
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    image_root = Path(args.image_root) if args.image_root else guess_image_root(raw_root)
    image_index = build_image_index(image_root)

    records_by_split = {'train': {}, 'dev': {}, 'test': {}}
    split_missing = {}

    if args.annotation_file:
        rows = read_table_annotations(args.annotation_file)
        for row in rows:
            if not isinstance(row, dict):
                continue

            image_ref = ''
            for key in IMAGE_KEYS:
                value = row.get(key)
                if value:
                    image_ref = value
                    break
            if not image_ref:
                continue

            image_rel_path = find_image_relpath(image_ref, image_index)
            if not image_rel_path:
                continue

            captions = extract_captions_from_row(row)
            if not captions:
                continue

            split = normalize_split_name(row.get('split') or row.get('subset'))
            if split:
                records_by_split.setdefault(split, {}).setdefault(image_rel_path, []).extend(captions)
            else:
                split_missing.setdefault(image_rel_path, []).extend(captions)
    else:
        for caption_path in iter_caption_texts(raw_root):
            image_rel_path = find_image_relpath(caption_path.stem, image_index)
            if not image_rel_path:
                continue

            caption = caption_path.read_text(encoding='utf-8', errors='ignore').strip()
            if not caption:
                continue

            split_missing.setdefault(image_rel_path, []).append(caption)

    if split_missing:
        derived_splits = split_records(
            split_missing,
            seed=args.seed,
            train_ratio=args.train_ratio,
            dev_ratio=args.dev_ratio,
        )
        for split, split_records_dict in derived_splits.items():
            for image_rel_path, captions in split_records_dict.items():
                records_by_split.setdefault(split, {}).setdefault(image_rel_path, []).extend(captions)

    write_retrieval_dataset(records_by_split, args.output_root)

    counts = {
        split: (len(records_by_split.get(split, {})), sum(len(v) for v in records_by_split.get(split, {}).values()))
        for split in ('train', 'dev', 'test')
    }
    print(json.dumps({
        'image_root': str(image_root.resolve()),
        'output_root': str(Path(args.output_root).resolve()),
        'counts': {
            split: {'images': image_count, 'captions': caption_count}
            for split, (image_count, caption_count) in counts.items()
        },
    }, indent=2))


if __name__ == '__main__':
    main()
