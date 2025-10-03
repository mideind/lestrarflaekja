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

    python prepare_data.py output_path=data/isl_debug.scramble subshard=1000 transform=scramble dataset_name=mideind/is_prototyping_corpus subset_names=mim,hugi output_path=data/isl_debug.scramble output_repoid=mideind/scramble.debug

    python prepare_data.py output_path=data/isl_debug.soup subshard=1000 transform=soup dataset_name=mideind/is_prototyping_corpus subset_names=mim,hugi output_path=data/isl_debug.soup output_repoid=mideind/soup.debug

    python prepare_data.py output_path=data/isl_debug.vanilla subshard=1000 transform=vanilla dataset_name=mideind/is_prototyping_corpus subset_names=mim,hugi output_path=data/isl_debug.vanilla output_repoid=mideind/vanilla.debug


    python prepare_data.py output_path=data/isl_debug.scramble subshard=1000 transform=scramble dataset_name=mideind/is_prototyping_corpus subset_names=blog.is,hugi,hugi,hugi,ic3v2,igc,mim,rafbokavefurinn,skemman,studentabladid  output_path=data/isl_debug.scramble output_repoid=mideind/scramble.debug
    python prepare_data.py output_path=data/isl_debug.soup subshard=1000 transform=soup dataset_name=mideind/is_prototyping_corpus subset_names=blog.is,hugi,hugi,hugi,ic3v2,igc,mim,rafbokavefurinn,skemman,studentabladid  output_path=data/isl_debug.soup output_repoid=mideind/soup.debug
    python prepare_data.py output_path=data/isl_debug.vanilla subshard=1000 transform=vanilla dataset_name=mideind/is_prototyping_corpus subset_names=blog.is,hugi,hugi,hugi,ic3v2,igc,mim,rafbokavefurinn,skemman,studentabladid output_path=data/isl_debug.vanilla output_repoid=mideind/vanilla.debug

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

import random
import logging
import os
import sys
from dataclasses import dataclass
from enum import StrEnum
from typing import Optional
import subprocess
from pathlib import Path
import tempfile

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


def mappable_transform_word_noise(batch, *, cfg, enc):
    """mappable transform

    batch is a dictof lists with keys=['text', 'text_clean', 'aux']
    """
    batch_size = len(batch["text"])

    list_text = batch.pop("text")
    list_text_clean = batch.pop("text_clean")
    list_aux = batch.pop("aux")

    assert len(batch) == 0

    batch["input_ids"] = []
    batch["weights"] = []

    for i in range(batch_size):
        text = list_text[i]
        text_clean = list_text[i]
        aux = list_aux[i]

        for chunk in chunk_text_by_word_count(
            text,
            min_words=cfg.min_words_main,
            max_words=cfg.max_words_main,
        ):
            if chunk is None:
                continue

            obj = transform_example_word_noise(
                text=chunk,
                text_clean=text_clean,
                cfg=cfg,
                enc=enc,
                text_aux=aux,
            )

            batch["input_ids"].append(obj["input_ids"])
            batch["weights"].append(obj["weights"])

    return batch


def mappable_transform_soup(batch, *, cfg, enc):
    """mappable transform

    batch is a dictof lists with keys=['text', 'text_clean', 'aux']
    """
    batch_size = len(batch["text"])

    list_text = batch.pop("text")
    list_text_clean = batch.pop("text_clean")
    list_aux = batch.pop("aux")

    assert len(batch) == 0

    batch["input_ids"] = []
    batch["weights"] = []

    for i in range(batch_size):
        text = list_text[i]
        text_clean = list_text[i]
        aux = list_aux[i]

        for chunk in chunk_text_by_word_count(
            text,
            min_words=cfg.min_words_main,
            max_words=cfg.max_words_main,
        ):
            if chunk is None:
                continue

            obj = transform_example_word_soup(
                text=chunk,
                text_clean=text_clean,
                cfg=cfg,
                enc=enc,
                text_aux=aux,
            )

            batch["input_ids"].append(obj["input_ids"])
            batch["weights"].append(obj["weights"])

    return batch


def prepare_dataset_word_noise(
    cfg: DataConfig, ds: hf_datasets.Dataset, *, enc: AutoTokenizer
) -> list[dict]:
    """process word noise."""
    logger.info("processing dataset for 'word-noise' task")
    augm_ds = normalize_and_make_auxiliary(cfg, ds)

    fn_kwargs = {
        "enc": enc,
        "cfg": cfg,
    }
    num_proc = 8 if len(augm_ds) > 1000 else 4
    ds = augm_ds.map(
        mappable_transform_word_noise,
        batched=True,
        batch_size=8,
        fn_kwargs=fn_kwargs,
        num_proc=num_proc,
    )

    return examples


def prepare_dataset_word_soup(
    cfg: DataConfig, ds: hf_datasets.Dataset, *, enc: AutoTokenizer
) -> list[dict]:
    """process word soup."""
    logger.info("processing dataset for 'word-soup' task")
    augm_ds = normalize_and_make_auxiliary(cfg, ds)

    fn_kwargs = {
        "enc": enc,
        "cfg": cfg,
    }
    num_proc = 8 if len(augm_ds) > 1000 else 4
    ds = augm_ds.map(
        mappable_transform_soup,
        batched=True,
        batch_size=8,
        fn_kwargs=fn_kwargs,
        num_proc=num_proc,
    )

    return ds


def prepare_dataset_vanilla(
    cfg: DataConfig, ds: hf_datasets.Dataset, *, enc: AutoTokenizer
) -> list[dict]:
    """process vanilla."""
    logger.info("processing dataset for 'vanilla' task")

    ds = ds.filter(
        lambda x: {"text": len(x["text"]) > cfg.coarse_prefilter_min_chars}
    )  # True means keep
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

    """
    if output_dir exists:
      end

    make output_dir

    for subset_idx and subset in subsets:
      out_subset ← process subset
      out_subset ← make train/test split in out_subset
      # save to disk
      ./output_dir/idx ← out_subset
      delete out_subset

    new_subsets ← list
    for subset_idx in subsets:
      new_subsets[idx] ← load subset_idx from disk

    epoch = concatenate new_subsets

    push epoch to hub
      
    """
    subset_names = [] if cfg.subset_names is None else cfg.subset_names.split(",")
    if subset_names:
        logger.info(f"dataset subset_names: {subset_names}")

    # no subsets ("configurations") provided
    if not subset_names:
        ds_dict = hf_datasets.load_dataset(cfg.dataset_name)
        out_splits = {}
        for split_name in ds_dict:
            ds = ds_dict[split_name]
            logger.info(f"loaded dataset: {len(ds)} examples")

            result_ds = preprocess_fns[cfg.transform](cfg, ds, enc=enc)
            out_splits[split_name] = result_ds

        merged_ds = hf_datasets.DatasetDict(out_splits)
        merged_ds.save_to_disk(str(cfg.output_path))

    else:
        # clear any old files if necessary and applicable
        if cfg.output_path.exists() and not cfg.overwrite:
            logger.info(f"Dataset already exists at '{cfg.output_path}'")
            sys.exit(0)
        if cfg.output_path.exists():
            _ret_code = subprocess.run(["rm", "-r", str(cfg.output_path)])

        # make sure our stuff exists
        cfg.output_path.mkdir(exist_ok=True, parents=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            _seed = cfg.seed
            for subset_idx, subset_name in enumerate(subset_names):
                s_ds = hf_datasets.load_dataset(cfg.dataset_name, name=subset_name)
                ds_train = s_ds.pop("train")
                ds_valid = s_ds.pop("test")

                logger.info(f"loaded subset '{subset_name}': {len(ds_train)} examples")

                cfg.seed = _seed + abs(hash(cfg.dataset_name + subset_name + str(subset_idx)))
                # subsample by sharding if requested
                if cfg.subshard is not None:
                    ds_train = ds_train.shard(cfg.subshard, 0)
                    logger.info(f"sharded '{subset_name}' to: {len(ds_train)} examples")

                result_ds_train = preprocess_fns[cfg.transform](cfg, ds_train, enc=enc)
                result_ds_valid = preprocess_fns[cfg.transform](cfg, ds_valid, enc=enc)

                logger.info(
                    f"output examples '{subset_name}': (train {len(result_ds_train):_d}) (valid {len(result_ds_valid):_d}) "
                )

                out_ds = hf_datasets.DatasetDict(
                    {
                        "train": result_ds_train,
                        "valid": result_ds_valid,
                    }
                )

                path_to_idx = Path(tmpdir) / f"subset.{subset_idx}"
                out_ds.save_to_disk(str(path_to_idx))
                del s_ds, ds_train, ds_valid, out_ds

            # load them all from disk and concatenate
            saved_datasets = []
            for subset_idx, subset_name in enumerate(subset_names):
                path_to_idx = Path(tmpdir) / f"subset.{subset_idx}"
                saved_ds = hf_datasets.load_from_disk(str(path_to_idx))
                saved_datasets.append(saved_ds)

            merged_ds = hf_datasets.DatasetDict(
                {
                    "train": hf_datasets.concatenate_datasets(
                        [s["train"] for s in saved_datasets]
                    ),
                    "valid": hf_datasets.concatenate_datasets(
                        [s["valid"] for s in saved_datasets]
                    ),
                }
            )
            merged_ds.save_to_disk(cfg.output_path)

            ntrain, nvalid = len(merged_ds["train"]), len(merged_ds["valid"])
            logger.info(
                f"Saved to '{cfg.output_path}': (train {ntrain:_d}) (valid {nvalid:_d}) examples"
            )

        # clear the tmpdir

    if cfg.output_repoid is not None:
        logger.info(f"pushing to huggingface hub: '{cfg.output_repoid}'")
        out_ds.push_to_hub(cfg.output_repoid)


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
