from dataclasses import dataclass
from enum import StrEnum

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
    tokenizer_name: str = "AI-Sweden-Models/gpt-sw3-126m"
    transform_type: ProcessType = ProcessType.WORD_NOISE
    noise_prob: float = 0.02
    permutation_distance: int = 2
    mask_token: str = "<mask>"
    soup_keep_rate: float = 0.5
    max_soup_words: int = 100
    max_prefix_words: int = 10
    bin_noise_low: float = 0.01
    bin_noise_medium: float = 0.025
    bin_noise_high: float = 0.05
    context_len: int = 512
    min_words_in_primary: int = 30
    max_words_in_primary: int = 128
    save_tokenized: bool = True
    delimiter: str = "<|endoftext|>"  # or any other delimiter you want to use
    use_hint: bool = True
    output_path: str = "data/word-noise"


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


def chunk_text_by_word_count(text: str, max_words: int, min_words: int) -> list[str]:
    """Chunk text by word count."""
    words = text.split()
    if len(words) < min_words:
        # discard too short texts
        return []
    text_parts = [words[i : i + max_words] for i in range(0, len(words), max_words)]
    # make sure the last segment is long enough
    if len(text_parts[-1]) < min_words:
        # discard last segment if it's too short
        text_parts.pop(-1)
    text_parts = [" ".join(part) for part in text_parts]
    return text_parts


def transform_example_word_noise(
    text: str, *, cfg: DataConfig, tokenizer: AutoTokenizer, aux: str
) -> dict:
    """Transform example with word noise.

    The following operations are applied in order:
    - delete
    - mask
    - insert
    - shuffle

    Assumin default hyperparameters:
    - The number of BPE tokens in the input_ids is approximately 2.2x the number of BPE tokens in the original text.
    - Alternatively, the number of BPE tokens is between 3x and 4x the number of words in the original text.
    """
    # split into wordlike tokens
    words = text.split()
    if len(words) < cfg.min_words_in_primary:
        return None

    # 1. sample which tokens to delete
    should_delete = np.random.uniform(0, 1, size=len(words)) < cfg.noise_prob
    should_keep = np.logical_not(should_delete)
    words = [word for word, keep in zip(words, should_keep) if keep]

    # 2. sample which tokens to mask
    should_mask = np.random.uniform(0, 1, size=len(words)) < cfg.noise_prob
    words = [cfg.mask_token if mask else word for word, mask in zip(words, should_mask)]

    # 3. sample when to insert token (using auxiliary example as source of tokens)
    should_insert = np.random.uniform(0, 1, size=len(words)) < cfg.noise_prob
    aux_words = aux.split()
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
    scramble = " ".join(words)

    # NOTE: we could make the within-k-shuffle respect sentence boundaries

    # hint for the task
    hint_str = f"[noise {bin_noise(cfg, cfg.noise_prob)}]"

    if not cfg.save_tokenized:
        return {
            "original": text,
            "scramble": scramble,
            "hint": hint_str,
            "delimiter": cfg.delimiter,
        }

    input_parts = [hint_str, cfg.delimiter] if cfg.use_hint else []
    input_parts.extend([scramble, cfg.delimiter])
    task_input = " ".join(input_parts)
    task_input = tokenizer(task_input, add_special_tokens=False)["input_ids"]
    task_output = tokenizer(text, add_special_tokens=False)["input_ids"]

    weights = [0] * len(task_input) + [1] * len(task_output)
    input_ids = task_input + task_output

    # fertility = len(tokenizer(example["text"], add_special_tokens=False)["input_ids"]) / len(words)
    # delim_ntoks = len(tokenizer(cfg.delimiter, add_special_tokens=False)["input_ids"])
    # delim_ntoks_space = len(tokenizer(" " + cfg.delimiter, add_special_tokens=False)["input_ids"])
    # hints_ntoks = len(tokenizer(" " + hint_str, add_special_tokens=False)["input_ids"])
    # hints_ntoks_space = len(tokenizer(" " + hint_str, add_special_tokens=False)["input_ids"])
    # scramble_ntoks = len(tokenizer(scramble, add_special_tokens=False)["input_ids"])
    # orig_ntoks = len(tokenizer(example["text"], add_special_tokens=False)["input_ids"])

    # summed = (
    #     delim_ntoks
    #     + delim_ntoks_space
    #     + hints_ntoks
    #     + hints_ntoks_space
    #     + scramble_ntoks
    #     + orig_ntoks
    # )

    # ic((
    #     delim_ntoks,
    #     delim_ntoks_space,
    #     hints_ntoks,
    #     hints_ntoks_space,
    #     scramble_ntoks,
    #     orig_ntoks,
    #     len(input_ids),
    #     summed,
    #     len(words),
    #     len(input_ids) / len(words),
    #     fertility,
    # ))
    # # ic((len(words), fertility, len(input_ids)))

    return {"input_ids": input_ids, "weights": weights}


def transform_example_word_soup(
    text: str, *, cfg: DataConfig, tokenizer: AutoTokenizer, aux_texts: list[str]
) -> dict:
    """Transform example with word soup."""
    # primary example, extract distinct words
    main_words = text.split()
    distinct_words = set(main_words)
    # drop some of the distinct words
    should_keep = np.random.uniform(0, 1, size=len(distinct_words)) < cfg.soup_keep_rate
    distinct_words = [word for (word, keep) in zip(distinct_words, should_keep) if keep]

    # auxiliary examples, same process as for primary example
    aux_stuff = []
    for aux_text in aux_texts:
        aux_words = set(aux_text.split())
        should_keep = np.random.uniform(0, 1, size=len(aux_words)) < cfg.soup_keep_rate
        aux_words = [word for (word, keep) in zip(aux_words, should_keep) if keep]
        aux_stuff.append(aux_words)

    soup_words = set()
    # combine distinct words from primary and auxiliary examples into list
    for aux_words in aux_stuff:
        soup_words.update(aux_words)
    soup_words.update(distinct_words)
    word_soup = list(soup_words)
    # shuffle list, and then truncate
    np.random.shuffle(word_soup)
    word_soup = word_soup[: cfg.max_soup_words]
    word_soup = " ".join(word_soup)

    # split primary example into prefix and rest (the reconstruction target)
    prefix = " ".join(main_words[: cfg.max_prefix_words])
    suffix = " ".join(main_words[cfg.max_prefix_words :])

    # extra hints for the task (binned noise rates, document count, etc.)
    hint_str = f"[noise {bin_noise(cfg, cfg.noise_prob)}] [docs {len(aux_texts)}]"

    if not cfg.save_tokenized:
        return {
            "original": text,
            "hint": hint_str,
            "prefix": prefix,
            "suffix": suffix,
            "word_soup": word_soup,
        }

    task_input_parts = [word_soup, cfg.delimiter, prefix, cfg.delimiter]
    if cfg.use_hint:
        task_input_parts = [hint_str, cfg.delimiter] + task_input_parts

    task_input = " ".join(task_input_parts)
    task_input = tokenizer(task_input, add_special_tokens=False)["input_ids"]
    task_output = tokenizer(suffix, add_special_tokens=False)["input_ids"]
    input_ids = task_input + task_output
    weights = [0] * len(task_input) + [1] * len(task_output)

    return {
        "input_ids": input_ids,
        "weights": weights,
    }


def encode_word_noise_task(cfg: TrainConfig, example: dict, tokenizer: AutoTokenizer) -> dict:
    """Tokenize word noise task."""
    input_parts = [example["hint"], cfg.delimiter] if cfg.use_hint else []
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


def tokenizer_fn(cfg: TrainConfig, example: dict, tokenizer: AutoTokenizer) -> dict:
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
