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

from utils import (
    DataConfig,
    ProcessType,
    transform_example_word_noise,
    transform_example_word_soup,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def prepare_dataset_word_noise(cfg: DataConfig, ds: hf_datasets.Dataset) -> None:
    """process word noise."""
    logger.info("process word noise")
    # primary iterator for examples, zipped with another iterator for auxiliary examples

    results = []
    for primary_example, aux_ex in zip(ds, ds):
        res_example = transform_example_word_noise(cfg, primary_example, aux=aux_ex)
        # ic(res_example)
        # breakpoint()
        results.append(res_example)

    # TODO: construct new dataset?
    return results


def prepare_dataset_word_soup(cfg: DataConfig, ds: hf_datasets.Dataset) -> None:
    """process word soup."""
    logger.info("process word soup")
    # primary iterator for examples, zipped with two auxiliary iterators

    results = []
    for primary_example, aux1, aux2 in zip(ds, ds, ds):
        res_example = transform_example_word_soup(cfg, primary_example, aux_examples=[aux1, aux2])
        # ic(res_example)
        # breakpoint()
        results.append(res_example)

    # TODO: construct new dataset?
    return results


def prepare_data(cfg: DataConfig) -> None:
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
