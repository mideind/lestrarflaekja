#!/bin/env python

import logging
from dataclasses import dataclass
from enum import StrEnum
from typing import Optional

import numpy as np
from transformers import (
    AutoTokenizer,
)


class ProcessType(StrEnum):
    """Process type."""

    WORD_NOISE = "word-noise"
    WORD_SOUP = "word-soup"


@dataclass
class DataConfig:
    """Configuration for the project."""

    # dataset_name: str = "mideind/mim"
    dataset_name: str = "mideind/mim-gold-21.05"
    # dataset_name: str = "mideind/is_prototyping_corpus"
    model_name: str = "AI-Sweden-Models/gpt-sw3-126m"
    process_type: ProcessType = ProcessType.WORD_NOISE
    noise_prob: float = 0.02
    permutation_distance: int = 2
    mask_token: str = "<mask>"
    soup_keep_rate: float = 0.5
    max_soup_words: int = 100
    max_soup_prefix_words: int = 10
    bin_noise_low: float = 0.01
    bin_noise_medium: float = 0.025
    bin_noise_high: float = 0.05
    context_len: int = 512
    tokenizer_name: str = "AI-Sweden-Models/gpt-sw3-126m"


@dataclass
class TrainConfig:
    """Configuration for the project."""

    # dataset_name: str = "mideind/mim"
    # dataset_name: str = "mideind/mim-gold-21.05"
    dataset_name: str = "vesteinn/babylm"
    # dataset_name: str = "mideind/is_prototyping_corpus"
    model_name: str = "AI-Sweden-Models/gpt-sw3-126m"
    context_len: int = 512
    delimiter: str = "<|endoftext|>"  # or any other delimiter you want to use


def bin_noise(cfg: DataConfig, noise_rate: float) -> str:
    """Bin noise rate."""
    if noise_rate < cfg.bin_noise_low:
        return "low"
    elif noise_rate < cfg.bin_noise_medium:
        return "medium"
    else:
        return "high"


def transform_example_word_noise(
    cfg: DataConfig, example: dict, *, aux: Optional[dict] = None
) -> dict:
    """Transform example with word noise.

    The following operations are applied in order:
    - delete
    - mask
    - insert
    - shuffle
    """
    # split into wordlike tokens
    words = example["text"].split()

    # 1. sample which tokens to delete
    should_delete = np.random.uniform(0, 1) < cfg.noise_prob
    should_keep = np.logical_not(should_delete)
    words = [word for word, keep in zip(words, should_keep) if keep]

    # 2. sample which tokens to mask
    should_mask = np.random.uniform(0, 1) < cfg.noise_prob
    words = [cfg.mask_token if mask else word for word, mask in zip(words, should_mask)]

    # 3. sample when to insert token (using auxiliary example as source of tokens)
    should_insert = np.random.uniform(0, 1) < cfg.noise_prob
    aux_words = aux["text"].split()
    # map to random word (insert of shuffling)
    perm = np.random.permutation(len(aux_words))
    words = [
        aux_words[perm[i]] if insert else word
        for i, (word, insert) in enumerate(zip(words, should_insert))
    ]

    # 4. sample which tokens to shuffle
    position_before_noise = np.arange(len(words), dtype=np.float32)
    position_noise = np.random.uniform(0, cfg.permutation_distance, size=len(words))
    position_after_noise = position_before_noise + position_noise
    words = [words[i] for i in np.argsort(position_after_noise)]

    # XXX: we could make the within-k-shuffle respect sentence boundaries

    # hint for the task
    hint_str = f"[noise {bin_noise(cfg, cfg.noise_prob)}]"

    # join words back into text
    result = {
        "original": example["text"],
        "scramble": " ".join(words),
        "hint": hint_str,
    }
    return result


def transform_example_word_soup(
    cfg: DataConfig, example: dict, *, aux_examples: list[dict]
) -> dict:
    """Transform example with word soup."""
    # primary example, extract distinct words
    main_words = example["text"].split()
    distinct_words = set(main_words)
    # drop some of the distinct words
    should_keep = np.random.uniform(0, 1, size=len(distinct_words)) < cfg.soup_keep_rate
    distinct_words = [word for (word, keep) in zip(distinct_words, should_keep) if keep]

    # auxiliary examples, same process as for primary example
    aux_stuff = []
    for aux_example in aux_examples:
        aux_words = set(aux_example["text"].split())
        should_keep = np.random.uniform(0, 1, size=len(aux_words)) < cfg.soup_keep_rate
        aux_words = [word for (word, keep) in zip(aux_words, should_keep) if keep]
        aux_stuff.append(aux_words)

    # combine distinct words from primary and auxiliary examples into list
    word_soup = list(distinct_words)
    for aux_words in aux_stuff:
        word_soup.extend(aux_words)
    # shuffle list, and then truncate
    np.random.shuffle(word_soup)
    word_soup = word_soup[: cfg.max_soup_words]
    word_soup = " ".join(word_soup)

    # split primary example into prefix and rest (the reconstruction target)
    prefix = " ".join(main_words[: cfg.max_soup_prefix_words])
    suffix = " ".join(main_words[cfg.max_soup_prefix_words :])

    # extra hints for the task (binned noise rates, document count, etc.)
    hint_str = f"[noise {bin_noise(cfg, cfg.noise_prob)}] [docs {len(aux_examples)}]"

    output = {
        "original": example["text"],
        "hint": hint_str,
        "prefix": prefix,
        "suffix": suffix,
        "word_soup": word_soup,
    }

    return output


def encode_word_noise_task(cfg: TrainConfig, example: dict, tokenizer: AutoTokenizer) -> dict:
    """Tokenize word noise task."""
    input_parts = [example["hint"], cfg.delimiter] if "hint" in example else []
    input_parts.extend([example["scramble"], cfg.delimiter])
    task_input = " ".join(input_parts)

    task_input = tokenizer(task_input, add_special_tokens=False)["input_ids"]
    task_output = tokenizer(example["original"], add_special_tokens=False)["input_ids"]
    input_ids = task_input + task_output
    weights = [0] * len(task_input) + [1] * len(task_output)

    return {"input_ids": input_ids, "loss_weights": weights}


def encode_word_soup_task(cfg: TrainConfig, example: dict, tokenizer: AutoTokenizer) -> dict:
    """Tokenize word soup task."""
    input_parts = [example["hint"], cfg.delimiter] if "hint" in example else []

    input_parts.extend([
        example["prefix"],
        cfg.delimiter,
        example["word_soup"],
        cfg.delimiter,
    ])
    task_input = " ".join(input_parts)

    task_input = tokenizer(task_input, add_special_tokens=False)["input_ids"]
    task_output = tokenizer(example["suffix"], add_special_tokens=False)["input_ids"]

    input_ids = task_input + task_output
    weights = [0] * len(task_input) + [1] * len(task_output)

    return {"input_ids": input_ids, "loss_weights": weights}


def tokenizer_fn( example: dict, tokenizer: AutoTokenizer, cfg: TrainConfig) -> dict:
    """Tokenize and pack sequences to minimize waste."""
    # Tokenize all texts
    all_tokens = []
    for text in example["text"]:
        tokens = tokenizer(text, add_special_tokens=False)["input_ids"]
        all_tokens.extend(tokens)
        all_tokens.append(tokenizer.eos_token_id)  # Add separator between texts

    # Segment into fixed-length sequences
    input_batch = []
    for i in range(0, len(all_tokens) - cfg.context_len + 1, cfg.context_len):
        input_batch.append(all_tokens[i : i + cfg.context_len])

    return {"input_ids": input_batch}
