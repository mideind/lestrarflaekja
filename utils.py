from dataclasses import dataclass
from enum import StrEnum
import re
from typing import Optional, NamedTuple
import functools
from pathlib import Path

from datasets import Dataset, concatenate_datasets
from omegaconf import MISSING
import numpy as np
from transformers import (
    AutoTokenizer,
)

PAT_ALPHANUMERIC = re.compile(r"[^\w ]", flags=re.UNICODE)
remove_non_alphanumeric = functools.partial(PAT_ALPHANUMERIC.sub, "")
PAT_MULTISPACE = re.compile(r"[ \t][ \t]+")
collapse_multispace = functools.partial(PAT_MULTISPACE.sub, " ")


class Transform(StrEnum):
    """Transform type."""

    scramble = "scramble"
    soup = "soup"


@dataclass
class DataConfig:
    """Configuration for the project."""

    subset_names: Optional[str] = None
    dataset_name: str = MISSING
    # dataset_name: str = "mideind/mim"
    # dataset_name: str = "mideind/mim-gold-21.05"
    # dataset_name: str = "mideind/is_prototyping_corpus"
    # subset_names=blog.is,hugi,hugi,hugi,ic3v2,igc,mim,rafbokavefurinn,skemman,studentabladid
    tokenizer_name: str = "AI-Sweden-Models/gpt-sw3-126m"
    transform: Transform = Transform.scramble
    permutation_distance: int = 5
    mask_token: str = "<mask>"
    soup_keep_rate: float = 0.66
    soup_ratio_low_bin: float = 0.25
    soup_ratio_med_bin: float = 0.5
    soup_ratio_high_bin: float = 0.75
    max_soup_words: int = 100
    max_prefix_words: int = 10
    scramble_low_bin: float = 0.01
    scramble_med_bin: float = 0.025
    scramble_high_bin: float = 0.05
    context_len: int = 512
    min_words_main: int = 30
    max_words_main: int = 128
    save_tokenized: bool = True
    delimiter: str = "<|endoftext|>"  # or any other delimiter you want to use
    use_hint: bool = True
    subshard: Optional[int] = None
    output_path: Path = MISSING
    seed: int = 42
    prefilter_char_count: int = 60


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


def nearest_bin_noise(
    cfg: DataConfig, noise_rate: float, low: float, med: float, high: float
) -> str:
    """Nearest bin label for noise rate."""
    arg = np.argmin(np.abs(noise_rate - np.array([low, med, high])))
    labels = ("low", "medium", "high")
    return labels[arg]


def chunk_text_by_word_count(text: str, min_words: int, max_words: int) -> list[str]:
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
    text: str, *, cfg: DataConfig, enc: AutoTokenizer, aux: str
) -> dict:
    """Transform example with word noise.

    The following operations are applied in order:
    - delete
    - mask
    - insert
    - shuffle

    The aux text is expected to be a cleaned version of the auxiliary text (no punctuation).

    Assuming default hyperparameters:
    - The number of BPE tokens in the input_ids is approximately 2.2x the number of BPE tokens in the original text.
    - Alternatively, the number of BPE tokens is between 3x and 4x the number of words in the original text.
    """
    # split into wordlike tokens
    words = text.split()
    if len(words) < cfg.min_words_main:
        return None

    scramble_rate = np.random.uniform(cfg.scramble_low_bin, cfg.scramble_high_bin)
    # 1. sample which tokens will be deleted
    should_delete = np.random.uniform(0, 1, size=len(words)) < scramble_rate
    should_keep = np.logical_not(should_delete)
    words = [word for word, keep in zip(words, should_keep) if keep]

    # 2. sample will be masked
    should_mask = np.random.uniform(0, 1, size=len(words)) < scramble_rate
    words = [cfg.mask_token if mask else word for word, mask in zip(words, should_mask)]

    # 3. sample the location of the inserted tokens
    should_insert = np.random.uniform(0, 1, size=len(words)) < scramble_rate
    aux_words = aux.split()
    # determine which words will be inserted
    perm = np.random.permutation(len(words)) % len(aux_words)
    # splice the words with randomly selected aux words
    spliced = []
    for i, (word, insert) in enumerate(zip(words, should_insert)):
        try:
            if insert:
                spliced.append(aux_words[perm[i]])
        except Exception as e:
            breakpoint()
        spliced.append(word)
    words = spliced

    # 4. sample new location after position of noise
    position_before_noise = np.arange(len(words), dtype=np.float32)
    position_noise = np.random.uniform(0, cfg.permutation_distance, size=len(words))
    position_after_noise = position_before_noise + position_noise

    words = [words[i] for i in np.argsort(position_after_noise)]
    scramble = " ".join(words)

    # hint for the task
    noise_bin = nearest_bin_noise(
        cfg, scramble_rate, cfg.scramble_low_bin, cfg.scramble_med_bin, cfg.scramble_high_bin
    )
    hint_str = f"[noise {noise_bin}]"

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
    task_input = enc(task_input, add_special_tokens=False)["input_ids"]
    task_output = enc(text, add_special_tokens=False)["input_ids"]

    weights = [0] * len(task_input) + [1] * len(task_output)
    input_ids = task_input + task_output

    return {"input_ids": input_ids, "weights": weights}


def transform_example_word_soup(
    text: str, *, cfg: DataConfig, enc: AutoTokenizer, clean_aux: str
) -> dict:
    """Transform example with word soup."""
    # extract just the words (without punctuation) from the text
    # cleaned_text = PAT_ALPHANUMERIC.sub("", text)
    cleaned_text = remove_non_alphanumeric(text)
    main_words = set(cleaned_text.split())
    # make sure some words are dropped
    should_keep = np.random.uniform(0, 1, size=len(main_words)) < cfg.soup_keep_rate
    kept_words = [word for (word, keep) in zip(main_words, should_keep) if keep]

    # distractor words from auxiliary texts
    aux_words = set(clean_aux.split())
    aux_words = [word for word in aux_words if word in main_words]
    should_keep = np.random.uniform(0, 1, size=len(aux_words)) < cfg.soup_keep_rate
    kept_aux = set([word for (word, keep) in zip(aux_words, should_keep) if keep])

    # convert to list and shuffle
    kept_aux = list(kept_aux)
    np.random.shuffle(kept_aux)

    # determine how much of the soup is a distraction
    soup_rate = np.random.uniform(cfg.soup_ratio_low_bin, cfg.soup_ratio_high_bin)
    num_primary_words = int(soup_rate * len(kept_words))
    num_aux_words = int((1 - soup_rate) * len(kept_words))
    soup_words = kept_words[:num_primary_words] + kept_aux[:num_aux_words]
    np.random.shuffle(soup_words)
    soup_words = soup_words[: cfg.max_soup_words]
    word_soup = " ".join(soup_words)

    # only the soup is punctuation-free
    whitespace_separated_tokens = text.split()
    # split the primary text into (prefix, suffix)
    prefix = " ".join(whitespace_separated_tokens[: cfg.max_prefix_words])
    # the suffix is the reconstruction target
    suffix = " ".join(whitespace_separated_tokens[cfg.max_prefix_words :])

    # we use one of 3 labels to hint at the noise rate (numbers don't work well in LMs)
    noise_bin = nearest_bin_noise(
        cfg, soup_rate, cfg.soup_ratio_low_bin, cfg.soup_ratio_med_bin, cfg.soup_ratio_high_bin
    )
    hint_str = f"[noise {noise_bin}] [docs {len(clean_aux)}]"

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
    task_input = enc(task_input, add_special_tokens=False)["input_ids"]
    task_output = enc(suffix, add_special_tokens=False)["input_ids"]
    input_ids = task_input + task_output
    weights = [0] * len(task_input) + [1] * len(task_output)

    return {
        "input_ids": input_ids,
        "weights": weights,
    }


class DatasetWithAuxiliary(NamedTuple):
    main: Dataset
    aux: Dataset

def normalize_and_make_auxiliary(cfg: DataConfig, ds: Dataset) -> DatasetWithAuxiliary:
    # drop obviosuly too short examples early
    ds = ds.filter(lambda x: {"text": len(x["text"]) > cfg.prefilter_char_count })  # True means keep

    ds_main = ds.map(lambda x: {"text": collapse_multispace(x["text"]).strip()})

    ds_main_clean = ds_main.map(lambda x: {"text": remove_non_alphanumeric(x["text"]).lower()})
    ds_clean = ds_main.map(lambda x: {"text": remove_non_alphanumeric(x["text"].lower())})

    # since these are shuffled, we don't need to shuffle ds_main as well
    ds_aux1 = ds_clean.shuffle(cfg.seed).rename_column("text", "text1")
    ds_aux2 = ds_aux1.shuffle(cfg.seed).rename_column("text1", "text2")
    # merge and flatten adjacent examples
    ds_aux = concatenate_datasets([ds_aux1, ds_aux2], axis=1)
    ds_aux = ds_aux.map(lambda x: {"text": x["text1"] + " " + x["text2"]})
    return DatasetWithAuxiliary(ds_main, ds_aux) 


def encode_word_noise_task(cfg: TrainConfig, example: dict, enc: AutoTokenizer) -> dict:
    """Tokenize word noise task."""
    input_parts = [example["hint"], cfg.delimiter] if cfg.use_hint else []
    input_parts.extend([example["scramble"], cfg.delimiter])
    task_input = " ".join(input_parts)

    task_input = enc(task_input, add_special_tokens=False)["input_ids"]
    task_output = enc(example["original"], add_special_tokens=False)["input_ids"]

    input_ids = task_input + task_output
    weights = [0] * len(task_input) + [1] * len(task_output)

    return {"input_ids": input_ids, "loss_weights": weights}


def encode_word_soup_task(cfg: TrainConfig, example: dict, enc: AutoTokenizer) -> dict:
    """Tokenize word soup task."""
    input_parts = [example["hint"], cfg.delimiter] if "hint" in example else []

    input_parts.extend([
        example["prefix"],
        cfg.delimiter,
        example["word_soup"],
        cfg.delimiter,
    ])
    task_input = " ".join(input_parts)

    task_input = enc(task_input, add_special_tokens=False)["input_ids"]
    task_output = enc(example["suffix"], add_special_tokens=False)["input_ids"]

    input_ids = task_input + task_output
    weights = [0] * len(task_input) + [1] * len(task_output)

    return {"input_ids": input_ids, "loss_weights": weights}


def tokenizer_fn(cfg: TrainConfig, example: dict, enc: AutoTokenizer) -> dict:
    """Tokenize and pack sequences to minimize waste."""
    # Tokenize all texts
    all_tokens = []
    for text in example["text"]:
        tokens = enc(text, add_special_tokens=False)["input_ids"]
        all_tokens.extend(tokens)
        all_tokens.append(enc.eos_token_id)  # Add separator between texts

    # Segment into fixed-length sequences
    input_batch = []
    for i in range(0, len(all_tokens) - cfg.context_len + 1, cfg.context_len):
        input_batch.append(all_tokens[i : i + cfg.context_len])

    return {"input_ids": input_batch}
