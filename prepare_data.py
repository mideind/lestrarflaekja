# pylint: disable=unused-import,unused-argument,W0611,logging-fstring-interpolation
# type: ignore[reportUnusedImport]
# ruff: noqa: F401

"""
1. velja gagnasafn til bráðabirgða
    ✓ https://huggingface.co/datasets/mideind/mim
    - https://huggingface.co/datasets/mideind/is_prototyping_corpus (kannski seinna)

2. velja líkan til bráðabirgða
    ✓ https://huggingface.co/AI-Sweden-Models/gpt-sw3-126m/
    - https://huggingface.co/AI-Sweden-Models/gpt-sw3-356m/ (kannski seinna)
    - https://huggingface.co/AI-Sweden-Models/gpt-sw3-1.3b/ (kannski enn síðar)

3. forrita gagnavarpanir
   a) endursmíði úr (orðabrengli)
   b) endursmíði úr (orðasúpu og forskeyti skjals)
   c) framkvæma endursmíðivarpanir

4. forrita þjálfunarskriftu sem notar (líkan, vörpuð gögn)

5. loss masking fyrir endursmíðiverkefnin

6. laga gagnadreifingu (auka vægi ákveðinna texta m. frumstæðum aðferðum)

7. setja af stað remote þjálfanir hjá RunPod af gerðunum (vanillu, brengl, súpu),
   ekki til að keyra á minni eigin tölvu.

8. fara heim
"""

import logging
import os
import sys
from dataclasses import dataclass
from enum import StrEnum
from typing import Optional

import datasets as hf_datasets
import numpy as np
from icecream import ic
from omegaconf import OmegaConf

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


class ProcessType(StrEnum):
    """Process type."""

    WORD_NOISE = "word-noise"
    WORD_SOUP = "word-soup"


@dataclass
class Config:
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
    soup_delim: str = "<|>"
    bin_noise_low: float = 0.01
    bin_noise_medium: float = 0.025
    bin_noise_high: float = 0.05


def bin_noise(cfg: Config, noise_rate: float) -> str:
    """Bin noise rate."""
    if noise_rate < cfg.bin_noise_low:
        return "low"
    elif noise_rate < cfg.bin_noise_medium:
        return "medium"
    else:
        return "high"


def transform_example_word_noise(cfg: Config, example: dict, *, aux: Optional[dict] = None) -> dict:
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

    # join words back into text
    example["text"] = " ".join(words)
    return example


def transform_example_word_soup(cfg: Config, example: dict, *, aux_examples: list[dict]) -> dict:
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

    # construct source and target (for the reconstruction task)
    source = f"{prefix} {cfg.soup_delim} {word_soup} {cfg.soup_delim} {suffix}"
    target = suffix
    as_contig_text = f"{source} {cfg.soup_delim} {target}"

    # extra hints for the task (binned noise rates, document count, etc.)
    hint_str = f"[noise: {bin_noise(cfg, cfg.noise_prob)}] [docs: {len(aux_examples)}]"
    as_contig_text = f"{hint_str} {as_contig_text}"

    output = {
        "text": as_contig_text,
        "hint": hint_str,
        "prefix": prefix,
        "suffix": suffix,
        "word_soup": word_soup,
        "delimiter": cfg.soup_delim,
    }

    return output


def prepare_dataset_word_noise(cfg: Config, ds: hf_datasets.Dataset) -> None:
    """process word noise."""
    logger.info("process word noise")
    # primary iterator for examples, zipped with another iterator for auxiliary examples

    results = []
    for primary_example, aux_ex in zip(ds, ds):
        res_example = transform_example_word_noise(cfg, primary_example, aux=aux_ex)
        # ic(res_example)
        # breakpoint()
        results.append(res_example)

    # XXX: construct new dataset?
    return results


def prepare_dataset_word_soup(cfg: Config, ds: hf_datasets.Dataset) -> None:
    """process word soup."""
    logger.info("process word soup")
    # primary iterator for examples, zipped with two auxiliary iterators

    results = []
    for primary_example, aux1, aux2 in zip(ds, ds, ds):
        res_example = transform_example_word_soup(cfg, primary_example, aux_examples=[aux1, aux2])
        # ic(res_example)
        # breakpoint()
        results.append(res_example)

    # XXX: construct new dataset?
    return results


def prepare_data(cfg: Config) -> None:
    """fooberino function."""
    logger.info("fooberino")
    dataset = hf_datasets.load_dataset(cfg.dataset_name, split="train")

    if cfg.process_type == ProcessType.WORD_NOISE:
        prepare_dataset_word_noise(cfg, dataset)
    elif cfg.process_type == ProcessType.WORD_SOUP:
        prepare_dataset_word_soup(cfg, dataset)
    else:
        assert False, "Invalid process type"


def main() -> None:
    """main function."""
    cfg = OmegaConf.structured(Config)
    cli_cfg = OmegaConf.from_cli()
    cfg = OmegaConf.merge(cfg, cli_cfg)
    cfg = OmegaConf.to_container(cfg, resolve=True)
    try:
        cfg = Config(**cfg)
    except TypeError as e:  # pylint: disable=broad-exception-raised
        logger.error(f"Error: {e}\n\nUsage: python scratch.py")
        sys.exit(1)
    prepare_data(cfg)


if __name__ == "__main__":
    main()
