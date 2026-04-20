#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from dataset_prep_common import (
    build_image_index,
    find_image_relpath,
    normalize_split_name,
    split_records,
    write_retrieval_dataset,
)


def extract_captions(item):
    captions = []

    if 'sentences' in item:
        for sentence in item['sentences']:
            if isinstance(sentence, str):
                captions.append(sentence)
            elif isinstance(sentence, dict):
                captions.append(
                    sentence.get('raw')
                    or sentence.get('caption')
                    or sentence.get('sent')
                    or sentence.get('text')
                    or ''
                )

    if 'captions' in item:
        for caption in item['captions']:
            if isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, dict):
                captions.append(caption.get('raw') or caption.get('caption') or caption.get('text') or '')

    return [caption for caption in captions if str(caption).strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation_json', required=True, help='path to dataset_rsicd.json')
    parser.add_argument('--image_root', required=True, help='path to the RSICD image directory')
    parser.add_argument('--output_root', default='data/rsicd', help='output directory in repository format')
    parser.add_argument('--seed', type=int, default=0, help='seed used when the annotations do not provide splits')
    args = parser.parse_args()

    annotation_path = Path(args.annotation_json)
    raw = json.loads(annotation_path.read_text(encoding='utf-8'))
    items = raw['images'] if isinstance(raw, dict) and 'images' in raw else raw

    image_index = build_image_index(args.image_root)
    records_by_split = {'train': {}, 'dev': {}, 'test': {}}
    split_missing = {}

    for item in items:
        if not isinstance(item, dict):
            continue

        image_ref = (
            item.get('filename')
            or item.get('file_name')
            or item.get('imgid')
            or item.get('image_id')
            or item.get('image')
            or ''
        )
        image_rel_path = find_image_relpath(image_ref, image_index)
        if not image_rel_path:
            continue

        captions = extract_captions(item)
        if not captions:
            continue

        split = normalize_split_name(item.get('split') or item.get('subset'))
        if split:
            records_by_split.setdefault(split, {})[image_rel_path] = captions
        else:
            split_missing[image_rel_path] = captions

    if split_missing:
        derived_splits = split_records(split_missing, seed=args.seed)
        for split, split_records_dict in derived_splits.items():
            records_by_split.setdefault(split, {}).update(split_records_dict)

    write_retrieval_dataset(records_by_split, args.output_root)

    counts = {
        split: (len(records_by_split.get(split, {})), sum(len(v) for v in records_by_split.get(split, {}).values()))
        for split in ('train', 'dev', 'test')
    }
    print(json.dumps({
        'output_root': str(Path(args.output_root).resolve()),
        'counts': {
            split: {'images': image_count, 'captions': caption_count}
            for split, (image_count, caption_count) in counts.items()
        },
    }, indent=2))


if __name__ == '__main__':
    main()
