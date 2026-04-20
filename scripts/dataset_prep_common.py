import csv
import json
import random
from pathlib import Path


IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


def normalize_text(text):
    return ' '.join(str(text).strip().split())


def normalize_split_name(split):
    split = (split or '').strip().lower()
    if split in {'train', 'training', 'trn', 'restval'}:
        return 'train'
    if split in {'val', 'valid', 'validation', 'dev'}:
        return 'dev'
    if split in {'test', 'testing', 'tst'}:
        return 'test'
    return ''


def iter_image_files(root):
    root = Path(root)
    for path in sorted(root.rglob('*')):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            yield path


def build_image_index(image_root):
    image_root = Path(image_root).resolve()
    by_name = {}
    by_stem = {}
    for image_path in iter_image_files(image_root):
        rel_path = image_path.relative_to(image_root).as_posix()
        by_name[image_path.name] = rel_path
        by_stem.setdefault(image_path.stem, rel_path)
    return image_root, by_name, by_stem


def find_image_relpath(image_ref, image_index):
    _, by_name, by_stem = image_index
    image_ref = str(image_ref).strip()
    if not image_ref:
        return ''

    ref_path = Path(image_ref)
    candidates = []
    if ref_path.suffix:
        candidates.append(ref_path.name)
    else:
        candidates.extend(ref_path.name + suffix for suffix in sorted(IMAGE_SUFFIXES))

    for candidate in candidates:
        if candidate in by_name:
            return by_name[candidate]

    if ref_path.stem in by_stem:
        return by_stem[ref_path.stem]

    return ''


def split_records(records, seed=0, train_ratio=0.8, dev_ratio=0.1):
    image_keys = sorted(records.keys())
    rng = random.Random(seed)
    rng.shuffle(image_keys)

    total = len(image_keys)
    n_train = int(total * train_ratio)
    n_dev = int(total * dev_ratio)

    split_to_keys = {
        'train': image_keys[:n_train],
        'dev': image_keys[n_train:n_train + n_dev],
        'test': image_keys[n_train + n_dev:],
    }
    return {
        split: {key: records[key] for key in keys}
        for split, keys in split_to_keys.items()
    }


def write_retrieval_dataset(records_by_split, output_root):
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    global_mapping = {}
    next_id = 0
    for split in ('train', 'dev', 'test'):
        for image_rel_path in sorted(records_by_split.get(split, {})):
            if image_rel_path not in global_mapping:
                global_mapping[image_rel_path] = next_id
                next_id += 1

    id_mapping = {str(image_id): rel_path for rel_path, image_id in global_mapping.items()}
    (output_root / 'id_mapping.json').write_text(json.dumps(id_mapping, indent=2, sort_keys=True) + '\n')

    for split in ('train', 'dev', 'test'):
        split_records = records_by_split.get(split, {})
        image_rel_paths = sorted(split_records)

        image_ids = [global_mapping[rel_path] for rel_path in image_rel_paths]
        captions = []
        capimgids = []

        for local_index, image_rel_path in enumerate(image_rel_paths):
            for caption in split_records[image_rel_path]:
                clean_caption = normalize_text(caption)
                if not clean_caption:
                    continue
                captions.append(clean_caption)
                capimgids.append(local_index)

        (output_root / f'{split}_ids.txt').write_text(
            ''.join(f'{image_id}\n' for image_id in image_ids)
        )
        (output_root / f'{split}_caps.txt').write_text(
            ''.join(f'{caption}\n' for caption in captions)
        )
        (output_root / f'{split}_capimgids.txt').write_text(
            ''.join(f'{capimgid}\n' for capimgid in capimgids)
        )


def read_table_annotations(path):
    path = Path(path)
    if path.suffix.lower() in {'.json', '.jsonl'}:
        text = path.read_text(encoding='utf-8')
        if path.suffix.lower() == '.jsonl':
            return [json.loads(line) for line in text.splitlines() if line.strip()]
        data = json.loads(text)
        if isinstance(data, dict) and 'images' in data:
            return data['images']
        if isinstance(data, list):
            return data
        raise ValueError(f'Unsupported JSON structure in {path}')

    delimiter = '\t' if path.suffix.lower() in {'.tsv', '.txt'} else ','
    with path.open('r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        return list(reader)
