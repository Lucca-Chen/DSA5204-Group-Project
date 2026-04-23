import random

import torch
from transformers import BertTokenizer


def get_tokenizer(opt):
    return BertTokenizer.from_pretrained(getattr(opt, 'bert_path', 'bert-base-uncased'))


def get_pad_token_id(tokenizer):
    pad_token_id = getattr(tokenizer, 'pad_token_id', None)
    if pad_token_id is None:
        return 0
    return int(pad_token_id)


def _process_caption_bert(tokenizer, caption, train=True, mask_rate=0.2, size_augment=True):
    tokens = tokenizer.basic_tokenizer.tokenize(caption)

    output_tokens = []
    deleted_idx = []

    for token in tokens:
        sub_tokens = tokenizer.wordpiece_tokenizer.tokenize(token)
        prob = random.random()

        if size_augment and prob < mask_rate and train:
            prob /= mask_rate
            if prob < 0.5:
                for _ in sub_tokens:
                    output_tokens.append("[MASK]")
            elif prob < 0.6:
                for _ in sub_tokens:
                    output_tokens.append(random.choice(list(tokenizer.vocab.keys())))
            else:
                for sub_token in sub_tokens:
                    output_tokens.append(sub_token)
                    deleted_idx.append(len(output_tokens) - 1)
        else:
            output_tokens.extend(sub_tokens)

    if deleted_idx:
        output_tokens = [output_tokens[i] for i in range(len(output_tokens)) if i not in deleted_idx]

    output_tokens = ['[CLS]'] + output_tokens + ['[SEP]']
    target = tokenizer.convert_tokens_to_ids(output_tokens)
    return torch.tensor(target, dtype=torch.long)


def process_caption(tokenizer, caption, opt, train=True):
    return _process_caption_bert(
        tokenizer,
        caption,
        train=train,
        size_augment=bool(getattr(opt, 'size_augment', 1)),
    )

