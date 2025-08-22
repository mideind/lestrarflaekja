# pylint: disable=unused-import,unused-argument,W0611,logging-fstring-interpolation
### type: ignore[reportUnusedImport]
### ruff: noqa: F401

"""
Data prep for task

────────────────────────────────────────────────────────────────────────────────

# Usage

make small soup:
    python prepare_data.py output_path=data/isl_desoup subshard=10 transform=soup dataset_name=mideind/is_prototyping_corpus subset_names=blog.is,hugi,hugi,hugi,ic3v2,igc,mim,rafbokavefurinn,skemman,studentabladid

make small scramble:
    python prepare_data.py output_path=data/isl_descramble subshard=10 transform=scramble dataset_name=mideind/is_prototyping_corpus subset_names=blog.is,hugi,hugi,hugi,ic3v2,igc,mim,rafbokavefurinn,skemman,studentabladid

────────────────────────────────────────────────────────────────────────────────

# Debug

    python prepare_data.py output_path=data/isl_debug.scramble subshard=1000 transform=scramble dataset_name=mideind/is_prototyping_corpus subset_names=mim,hugi output_path=data/isl_debug.scramble

    python prepare_data.py output_path=data/isl_debug.vanilla subshard=1000 transform=vanilla dataset_name=mideind/is_prototyping_corpus subset_names=mim,hugi output_path=data/isl_debug.vanilla

────────────────────────────────────────────────────────────────────────────────

# Pushing
    
scramble:
    python prepare_data.py output_repoid=mideind/scramble.debug output_path=data/isl_debug.scramble subshard=1000 transform=scramble dataset_name=mideind/is_prototyping_corpus subset_names=mim,hugi,hugi

soup:
    python prepare_data.py output_repoid=mideind/soup.debug output_path=data/isl_debug.soup subshard=1000 transform=soup dataset_name=mideind/is_prototyping_corpus subset_names=mim,hugi,hugi

local:
    python push_local_to_hub.py  local_path=data/isl_debug.soup  repoid=mideind/soup.debug

────────────────────────────────────────────────────────────────────────────────

fooscratch

python prepare_data.py \
    output_path=data/isl_descramble \
    transform=scramble \
    subshard=1000 \
    dataset_name=mideind/is_prototyping_corpus \
    subset_names=blog.is,hugi,hugi,hugi,ic3v2,igc,mim,rafbokavefurinn,skemman,studentabladid

"""


import logging
import os
import sys
from dataclasses import dataclass
from enum import StrEnum
from typing import Optional

import datasets as hf_datasets
from datasets import concatenate_datasets
import numpy as np
import tqdm
from icecream import ic
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from utils import (
    DataConfig,
    Transform,
    chunk_text_by_word_count,
    transform_example_word_noise,
    transform_example_word_soup,
    transform_vanilla,
    PAT_ALPHANUMERIC,
    remove_non_alphanumeric,
    PAT_MULTISPACE,
    collapse_multispace,
    normalize_and_make_auxiliary,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def prepare_dataset_word_noise(
    cfg: DataConfig, ds: hf_datasets.Dataset, *, enc: AutoTokenizer
) -> list[dict]:
    """process word noise."""
    logger.info("processing dataset with word noise")

    augm_ds = normalize_and_make_auxiliary(cfg, ds)

    examples = []
    for doc, aux in tqdm.tqdm(zip(augm_ds.main, augm_ds.aux), total=len(augm_ds)):
        # make sure the document is not too short and not too long by partitioning
        chunks = chunk_text_by_word_count(
            doc["text"], min_words=cfg.min_words_main, max_words=cfg.max_words_main
        )

        # we reuse the same aux document for examples derived from same primary document
        for chunk in chunks:
            result = transform_example_word_noise(
                text=chunk,
                cfg=cfg,
                enc=enc,
                aux=aux["text"],
            )

            if result is None:
                continue
            examples.append(result)

    return examples


def prepare_dataset_word_soup(
    cfg: DataConfig, ds: hf_datasets.Dataset, *, enc: AutoTokenizer
) -> list[dict]:
    """process word soup."""
    logger.info("processing dataset with word soup")

    augm_ds = normalize_and_make_auxiliary(cfg, ds)

    examples = []
    for doc, aux in tqdm.tqdm(zip(augm_ds.main, augm_ds.aux), total=len(augm_ds)):

        chunks = chunk_text_by_word_count(
            doc["text"], min_words=cfg.min_words_main, max_words=cfg.max_words_main
        )

        for chunk in chunks:
            result = transform_example_word_soup(
                text=chunk,
                cfg=cfg,
                enc=enc,
                clean_aux=aux["text"],
            )
            if result is None:
                continue
            examples.append(result)

    return examples

def prepare_dataset_vanilla(
    cfg: DataConfig, ds: hf_datasets.Dataset, *, enc: AutoTokenizer
) -> list[dict]:
    """process vanilla."""
    logger.info("processing dataset with vanilla")

    ds = ds.filter(lambda x: {"text": len(x["text"]) > cfg.prefilter_char_count })  # True means keep
    ds = ds.map(lambda x: {"text": collapse_multispace(x["text"]).strip()})

    examples = []
    for doc in tqdm.tqdm(ds, total=len(ds)):

        chunks = chunk_text_by_word_count(
            doc["text"], min_words=cfg.min_words_main, max_words=cfg.max_words_main
        )

        for chunk in chunks:
            result = transform_vanilla(
                text=chunk,
                cfg=cfg,
                enc=enc,
            )
            if result is None:
                continue
            examples.append(result)

    return examples

def prepare_data(cfg: DataConfig) -> None:
    """The fooberino."""
    logger.info(f"loading tokenizer: {cfg.tokenizer_name}")
    enc = AutoTokenizer.from_pretrained(cfg.tokenizer_name)
    logger.info(f"loading dataset: {cfg.dataset_name}")

    logger.info(f"task type: {cfg.transform.value}")
    preprocess_fns = {
        Transform.scramble: prepare_dataset_word_noise,
        Transform.soup: prepare_dataset_word_soup,
        Transform.vanilla: prepare_dataset_vanilla,
    }

    subset_names = [] if cfg.subset_names is None else cfg.subset_names.split(",")
    if subset_names:
        logger.info(f"dataset subset_names: {subset_names}")

    # no subsets to think of
    if not subset_names:
        ds = hf_datasets.load_dataset(cfg.dataset_name, split="train")
        logger.info(f"loaded dataset: {len(dataset)} examples")
        examples = preprocess_fns[cfg.transform](cfg, ds, enc=enc)
        logger.info(f"──────────────────────────────")

    else:
        # we want distractors to be similar to reconstruction target
        # therefore each subset has to be handled individually
        # instead of combining all into one first
        examples = []
        for name in subset_names:
            subset_ds = hf_datasets.load_dataset(cfg.dataset_name, name=name, split="train")

            # subsample by sharding if requested
            if cfg.subshard is not None:
                subset_ds = subset_ds.shard(cfg.subshard, 0)

            logger.info(f"loaded subset '{name}': {len(subset_ds)} examples")

            subset_examples = preprocess_fns[cfg.transform](cfg, subset_ds, enc=enc)
            logger.info(f"total examples from '{name}': {len(subset_examples)}")
            logger.info(f"──────────────────────────────")

            examples.extend(subset_examples)

    logger.info(f"total examples: {len(examples)}")
    logger.info(f"saving dataset to: {cfg.output_path}")

    out_ds = hf_datasets.Dataset.from_list(examples)
    out_ds.save_to_disk(str(cfg.output_path))

    if cfg.output_repoid is not None:
        logger.info(f"pushing to huggingface hub: '{cfg.output_repoid}'")
        out_ds.push_to_hub(cfg.output_repoid)

    return out_ds


def main() -> None:
    """main function."""
    cfg = OmegaConf.structured(DataConfig)
    cli_cfg = OmegaConf.from_cli()
    cfg = OmegaConf.merge(cfg, cli_cfg)
    try:
        cfg = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        cfg = DataConfig(**cfg)
    except Exception as e:  # pylint: disable=broad-exception-raised
        ic(cfg)
        raise e
    prepare_data(cfg)


if __name__ == "__main__":
    main()
