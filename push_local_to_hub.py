
from dataclasses import dataclass
from enum import StrEnum
import re
from typing import Optional, NamedTuple
import functools
from pathlib import Path

from datasets import Dataset, concatenate_datasets, load_dataset, load_from_disk
from omegaconf import MISSING, OmegaConf
from icecream import ic

@dataclass
class PushLocalConfig:
    """Configuration."""

    local_path: Path = MISSING
    repoid: str = MISSING


def load_and_push(cfg: PushLocalConfig):
    ds = load_from_disk(cfg.local_path)
    breakpoint()
    ds.push_to_hub(cfg.repoid)


def main() -> None:
    """main function."""
    cfg = OmegaConf.structured(PushLocalConfig)
    cli_cfg = OmegaConf.from_cli()
    cfg = OmegaConf.merge(cfg, cli_cfg)
    try:
        cfg = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        cfg = PushLocalConfig(**cfg)
    except Exception as e:  # pylint: disable=broad-exception-raised
        ic(cfg)
        raise e
    load_and_push(cfg)


if __name__ == "__main__":
    main()
