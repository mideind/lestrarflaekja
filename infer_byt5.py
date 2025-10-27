import torch

import logging
import functools
from dataclasses import dataclass
from typing import Any

from loguru import logger
import numpy
import torch
import datasets as hf_datasets
from omegaconf import OmegaConf
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)
from transformers import Trainer, TrainingArguments
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import PeftModel, LoraConfig, get_peft_model, prepare_model_for_kbit_training
from accelerate import Accelerator
from icecream import ic

example_texts = [
    """Heit og geislandi uppnumin í sæluvímu. Ást og aðdáun allra þeirra sem nutu hennar mettaði sál hennar dýrðlegri fullnægju, sem sökk niður í djúp vit­undarinnar. Fullnægju sem varð að óslökkvandi, já óseðjandi þrá,sem hvatti hana til dáða.Fékk hana til að skína og skína. Já, hvort hún skyldi skína í allri sinni dýrð!

Skyndilega féll skuggi á þessa dásamlegu stund. Það fór hreinlega um hana. Já, hún skalf við tilhugsunina. Skyndilega varð henni ískalt, þrátt fyrir að henni væri alltaf heitt í hamsi. Auðvitað gat hann ekki látið hana í friði. Þessi andstyggilegi ísmeygilegi andstæðingur. Hann hafði, svo lengi sem hana rak minni til , reynt að gera lítið úr henni. Hún vissi alveg hvað fór svona í taugarnar á honum. Ó,já hann var hreinlega að sturlast úr öfund. Hún vissi nákvæmlega hvað hann þráði heitt þá ást og aðdáun sem hún var aðnjótandi. Guð hafði svo sannarlega haft á réttu að standa þegar hann fullyrti að langdregin þrá geri hjartað sjúkt. Hún hafði alltaf fyrirlitið þennan óvildarmann sinn en kannski var honum bara vorkunn."""
]


@dataclass
class InferConfig:
    """Configuration for the project."""

    # model_id: str = "mideind/byt5-large-spanmask-pretrain"
    model_id: str = "mideind/byt5-large-spanmask-pretrain"
    # legal values are 1-30
    mask_length: int = 25
    allow_span_length_hints: bool = False
    # discard the scores of bytes near the edges of the target span,
    # since they are typically partially revealed words
    margin: int = 0


@dataclass
class ScoredExample:
    pass


@dataclass
class ScoredChunk:
    start: int
    end: int
    scores: torch.Tensor
    # byte_ids: torch.Tensor


@dataclass
class ExampleScorer:
    cfg: InferConfig
    model: AutoModelForSeq2SeqLM
    tokenizer: AutoTokenizer

    @classmethod
    def from_config(cls, cfg: InferConfig):
        byte_tokenizer = AutoTokenizer.from_pretrained("google/byt5-small")
        # TODO: device map
        model = AutoModelForSeq2SeqLM.from_pretrained(cfg.model_id)
        return cls(cfg=cfg, model=model, tokenizer=byte_tokenizer)

    def score_string(self, text: str):
        # we are assuming no overlap (stride=mask_length) for now
        # the mask sequence without hints is "<MASK>" (in upper case),
        # the mask sequence with length hint is f"<MASK_{length}>"
        logger.info(text)
        words = text.split()
        byte_ids = torch.tensor(self.tokenizer(text).input_ids)
        # (T) → (B × T)
        byte_ids = byte_ids.unsqueeze(0)
        unhinted_mask = "<MASK>"

        mask_seq = torch.tensor(self.tokenizer("<MASK>").input_ids)
        logger.debug(mask_seq.shape)

        idxs = list(range(0, len(byte_ids), self.cfg.mask_length // 2))
        # add end point of last interval
        if idxs[-1] < len(byte_ids) - 5:
            idxs.append(len(byte_ids))

        intervals = list(zip(idxs[:-1], idxs[1:]))

        accums = torch.zeros_like(byte_ids, dtype=torch.float)
        denoms = torch.zeros_like(accums)
        scored_chunks = []

        for chunk_idx, (mask_start, mask_end) in enumerate(intervals):
            byte_ids_with_mask = torch.cat(
                [byte_ids[:mask_start], mask_seq, byte_ids[mask_end:]]
            )
            logger.debug(byte_ids_with_mask.shape)
            # (T) → (B × T)
            byte_ids_with_mask = byte_ids_with_mask.unsqueeze(0)

            out = model(input_ids=byte_ids_with_mask, labels=byte_ids)
            logger.debug(out.logits.shape)
            target_seq = byte_ids_[mask_start:mask_end]

            chunk_scores = out.logits.squeeze(0)[mask_start:mask_end].cpu()
            chunk_scores = chunk_scores.gather(index=byte_ids, dim=2)
            chunk = ScoredChunk(start=mask_start, end=mask_end, scores=chunk_scores)

            scored_chunks.append(scored_chunk)
            accums[mask_start:mask_end] += chunk.scores
            denoms[mask_start:mask_end] += 1.0

        # prevent division by zero
        denoms[denoms.eq(0)] = 1

        # TODO: we might want to discard scores near the edges of target span
        final_score = accums / denoms

        pass
        # rich.print()


def do_main(cfg: InferConfig):
    scorer = ExampleScorer.from_config(cfg=cfg)
    result = scorer.score_string(example_texts[0])


def main() -> None:
    """main function"""
    cfg = OmegaConf.structured(InferConfig)
    cli_cfg = OmegaConf.from_cli()
    cfg = OmegaConf.merge(cfg, cli_cfg)
    cfg = OmegaConf.to_container(cfg, resolve=True)
    try:
        cfg = InferConfig(**cfg)
    except TypeError as e:  # pylint: disable=broad-exception-raised
        logger.error(f"Error: {e}\n\nUsage: python scratch.py")
        sys.exit(1)

    do_main(cfg)


if __name__ == "__main__":
    main()
