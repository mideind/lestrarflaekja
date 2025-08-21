# pylint: disable=unused-import,unused-argument,W0611,logging-fstring-interpolation
# type: ignore[reportUnusedImport]
# ruff: noqa: F401

"""
──────────────────────────────────────────────────────────────────────────────
# Dagur 1:

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

──────────────────────────────────────────────────────────────────────────────
# Dagur 2:

1. vantar að klippa skjölin í búta 512 orða búta
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
from transformers import AutoTokenizer

from utils import (
    DataConfig,
    ProcessType,
    chunk_text_by_word_count,
    transform_example_word_noise,
    transform_example_word_soup,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def prepare_dataset_word_noise(
    cfg: DataConfig, ds: hf_datasets.Dataset, *, tokenizer: AutoTokenizer
) -> hf_datasets.Dataset:
    """process word noise."""
    logger.info("processing dataset with word noise")

    output_examples = []
    for document, aux_ex in zip(ds, ds):
        # make sure the document is not too short and not too long by partitioning
        chunks = chunk_text_by_word_count(
            document["text"], cfg.max_words_in_primary, cfg.min_words_in_primary
        )

        # we reuse the same aux document for examples derived from same primary document
        for chunk in chunks:
            result = transform_example_word_noise(
                text=chunk,
                cfg=cfg,
                tokenizer=tokenizer,
                aux=aux_ex["text"],
            )

            if result is None:
                continue
            output_examples.append(result)

    # construct new dataset
    out_ds = hf_datasets.Dataset.from_list(output_examples)
    logger.info(f"constructed dataset: {len(out_ds)} examples")
    return out_ds


def prepare_dataset_word_soup(
    cfg: DataConfig, ds: hf_datasets.Dataset, *, tokenizer: AutoTokenizer
) -> hf_datasets.Dataset:
    """process word soup."""
    logger.info("processing dataset with word soup")

    output_examples = []
    for document, aux1, aux2 in zip(ds, ds, ds):
        chunks = chunk_text_by_word_count(
            document["text"], cfg.max_words_in_primary, cfg.min_words_in_primary
        )

        for chunk in chunks:
            result = transform_example_word_soup(
                text=chunk,
                cfg=cfg,
                tokenizer=tokenizer,
                aux_texts=[aux1["text"], aux2["text"]],
            )
            if result is None:
                continue
            output_examples.append(result)

    # construct new dataset
    out_ds = hf_datasets.Dataset.from_list(output_examples)
    logger.info(f"constructed dataset: {len(out_ds)} examples")
    return out_ds


def prepare_data(cfg: DataConfig) -> None:
    """fooberino function."""
    logger.info("loading dataset")
    dataset = hf_datasets.load_dataset(cfg.dataset_name, split="train")
    logger.info(f"loaded dataset: {len(dataset)} examples")

    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name)
    logger.info("tokenizer loaded")

    if cfg.transform_type == ProcessType.WORD_NOISE:
        out_ds = prepare_dataset_word_noise(cfg, dataset, tokenizer=tokenizer)
    elif cfg.transform_type == ProcessType.WORD_SOUP:
        out_ds = prepare_dataset_word_soup(cfg, dataset, tokenizer=tokenizer)
    else:
        assert False, "Invalid process type"

    logger.info(f"saving dataset to {cfg.output_path}")
    # save dataset
    out_ds.save_to_disk(cfg.output_path)


def main() -> None:
    """main function."""
    cfg = OmegaConf.structured(DataConfig)
    cli_cfg = OmegaConf.from_cli()
    cfg = OmegaConf.merge(cfg, cli_cfg)
    cfg = OmegaConf.to_container(cfg, resolve=True)
    try:
        cfg = DataConfig(**cfg)
    except TypeError as e:  # pylint: disable=broad-exception-raised
        logger.error(f"Error: {e}\n\nUsage: python scratch.py")
        sys.exit(1)
    prepare_data(cfg)


if __name__ == "__main__":
    main()
