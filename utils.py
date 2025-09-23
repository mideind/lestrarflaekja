from dataclasses import dataclass
from enum import StrEnum
import re
from typing import Optional, NamedTuple
import functools
from pathlib import Path

from datasets import Dataset, concatenate_datasets, load_from_disk
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
    vanilla = "vanilla"


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
    output_repoid: Optional[str] = None
    seed: int = 42
    coarse_prefilter_min_chars: int = 60


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


def transform_vanilla(text: str, *, cfg: DataConfig, enc: AutoTokenizer) -> dict:
    """vanilla."""

    task_input = enc(text, add_special_tokens=False)["input_ids"]
    return {
        "input_ids": task_input,
    }


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
        cfg,
        scramble_rate,
        cfg.scramble_low_bin,
        cfg.scramble_med_bin,
        cfg.scramble_high_bin,
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
    text: str, *, text_clean, text_aux: str, cfg: DataConfig, enc: AutoTokenizer
) -> dict:
    """Transform example with word soup."""
    src_words = set(text_clean.split())
    # make sure some words are dropped
    should_keep = np.random.uniform(0, 1, size=len(src_words)) < cfg.soup_keep_rate
    kept_src_words = [word for (word, keep) in zip(src_words, should_keep) if keep]

    # distractor words from auxiliary texts
    distractors = set(text_aux.split())
    distractors = [word for word in distractors if word in src_words]
    should_keep = np.random.uniform(0, 1, size=len(distractors)) < cfg.soup_keep_rate
    kept_distractors = set(
        [word for (word, keep) in zip(distractors, should_keep) if keep]
    )

    # convert to list and shuffle
    kept_distractors = list(kept_distractors)
    np.random.shuffle(kept_distractors)

    # determine how much of the soup is a distraction
    soup_ratio = np.random.uniform(cfg.soup_ratio_low_bin, cfg.soup_ratio_high_bin)
    # we mix part of the source document...
    num_src_words = int(soup_ratio * len(kept_src_words))
    # with some distractors...
    num_distractors = int((1 - soup_ratio) * len(kept_src_words))
    # to make a soup
    the_soup = kept_src_words[:num_src_words] + kept_distractors[:num_distractors]
    # mix the soup
    np.random.shuffle(the_soup)
    # shouldn't be necessary, but just in case
    the_soup = the_soup[: cfg.max_soup_words]
    # bake the soup
    word_soup = " ".join(the_soup)

    # the target text is not normalized and still may have punctuation and such,
    # we want to put some of it as an orderly side plate next to the soup without mixing it
    whitespace_separated_tokens = text.split()
    # separate the source document into (prefix, suffix)
    # the prefix is and the soup are the inputs for the task
    prefix = " ".join(whitespace_separated_tokens[: cfg.max_prefix_words])
    # the task is to "reconstruct" the suffix (so they are the targets)
    suffix = " ".join(whitespace_separated_tokens[cfg.max_prefix_words :])

    # we use one of 3 labels to hint at the noise rate (numbers don't work well in LMs)
    noise_bin = nearest_bin_noise(
        cfg,
        soup_ratio,
        cfg.soup_ratio_low_bin,
        cfg.soup_ratio_med_bin,
        cfg.soup_ratio_high_bin,
    )
    hint_str = f"[noise {noise_bin}]"

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


def normalize_clone_clean(example: dict) -> dict:
    text = collapse_multispace(example["text"]).strip()
    text_clean = remove_non_alphanumeric(text).lower()
    return {"text": text, "text_clean": text_clean}


def normalize_and_make_auxiliary(cfg: DataConfig, ds: Dataset) -> DatasetWithAuxiliary:
    # drop obviosuly too short examples early (True means keep example in dataset)
    ds = ds.filter(lambda x: {"text": len(x["text"]) > cfg.coarse_prefilter_min_chars})
    ds = ds.map(normalize_clone_clean)
    ds = ds.filter(lambda x: {"text": len(x["text"]) > cfg.coarse_prefilter_min_chars})

    # save to disk to free memory
    ds.save_to_disk(f"{cfg.output_path}.tmp")
    del ds
    ds_main = load_from_disk(f"{cfg.output_path}.tmp")

    # we need two streams of auxiliary examples,
    # they are used as the source of noise when adding noise
    # to the proper (main) example
    ds_aux = ds_main.shuffle(cfg.seed + 42)
    # we only need the normalized cleaned text of the auxiliaries
    unneeded_columns = [col for col in ds_aux.column_names if "text_clean" != col]
    ds_aux = ds_aux.remove_columns(unneeded_columns)
    ds_aux = ds_aux.rename_column("text_clean", "aux")

    # make a copy
    ds_aux_other = ds_aux.rename_column("aux", "aux_other")
    # make aux and its copy shuffled relative to each other
    ds_aux = ds_aux.shuffle(cfg.seed + 1337)

    # combine them horizontally
    ds_aux = concatenate_datasets([ds_aux, ds_aux_other], axis=1)
    # flatten them into one string
    ds_aux = ds_aux.map(lambda x: {"aux": x["aux"] + " " + x["aux_other"]})
    ds_aux = ds_aux.remove_columns(["aux_other"])

    # shuffle main so that the three (main, aux, aux_other)
    # originate from three independently sampled examples
    ds_aux = ds_aux.shuffle(cfg.seed + 1338)

    # merge aux with the main horizontally
    out_ds = concatenate_datasets([ds_main, ds_aux], axis=1)
    return out_ds


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

    input_parts.extend(
        [
            example["prefix"],
            cfg.delimiter,
            example["word_soup"],
            cfg.delimiter,
        ]
    )
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
