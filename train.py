"""
1. Sækja huggingface dataset

2. sækja huggingface model
    - https://huggingface.co/AI-Sweden-Models/gpt-sw3-126m/

3. Þjálfa á ["text"] lyklinum, vanilla auto-regressive transformer
- en geta breytt og fiktað í lossinum
"""

import logging
import functools
from dataclasses import dataclass

import torch
import datasets as hf_datasets
from omegaconf import OmegaConf
from transformers import (
    AutoModelForCausalLM,
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


accelerator = Accelerator()

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)
NAT_LOG_OF_2 = 0.6931471805599453


@dataclass
class Config:
    """Configuration for the project."""

    # dataset_name: str = "mideind/mim"
    # dataset_name: str = "mideind/mim-gold-21.05"
    dataset_name: str = "vesteinn/babylm"
    # dataset_name: str = "mideind/is_prototyping_corpus"
    # model_name: str = "AI-Sweden-Models/gpt-sw3-126m"
    model_name: str = "AI-Sweden-Models/gpt-sw3-356m"
    # model_name: str = "AI-Sweden-Models/gpt-sw3-1.3b"
    batch_size: int = 32
    accumulate_steps: int = 1
    warmup_steps: int = 100
    use_lora: bool = False
    lora_dropout: float = 0.05
    lora_rank: int = 32
    lora_alpha: int = 16
    logging_steps: int = 5
    max_steps: int = 1000
    eval_steps: int = 100
    save_steps: int = 1000
    learning_rate: float = 5e-5

    # eval_steps=cfg.eval_steps,
    # logging_steps=cfg.logging_steps,
    # gradient_accumulation_steps=cfg.accumulate_steps,
    # num_train_epochs=1,
    # weight_decay=0.01,
    # warmup_steps=cfg.warmup_steps,
    # lr_scheduler_type="cosine",
    # save_steps=cfg.save_steps,
    # max_steps=cfg.max_steps,
    # bf16=True,
    # push_to_hub=False,
    # label_names=["labels", "weights"],


class ReconstructionTaskCollator(DataCollatorForLanguageModeling):
    """Collator for the reconstruction task."""

    def __init__(self, tokenizer):
        super().__init__(tokenizer, mlm=False)

    def torch_call(self, examples: list[dict]) -> dict:
        # the super method does not handle our dict keys

        assert "weights" in examples[0]

        # Handle dict or lists with proper padding and conversion to tensor.

        if self.seed and self.generator is None:
            # If we have a seed, we need to create a generator object. Subsequent calls to this function will use the same generator.
            # If no seed supplied, we will use the global RNG
            self.create_rng()

        input_ids = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(example["input_ids"]) for example in examples],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )

        # (B × T)
        input_mask = input_ids.ne(self.tokenizer.pad_token_id)
        # (B × T × 1) · (B × 1 × T) → (B × T × T)
        input_mask = input_mask.unsqueeze(-1) @ input_mask.unsqueeze(1)
        # (B × T × T)
        attention_mask = input_mask.tril()

        # shift the input so we predict the next token
        labels = input_ids.roll(-1)
        labels[:, -1] = self.tokenizer.pad_token_id
        labels[labels.eq(self.tokenizer.pad_token_id)] = -100

        if "weights" in examples[0]:
            weights = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(example["weights"]) for example in examples],
                batch_first=True,
                padding_value=0,
            ).float()

            return {
                "input_ids": input_ids,
                "labels": labels,
                "weights": weights,
                "attention_mask": attention_mask,
            }

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }


class TruncatedLossTrainer(Trainer):
    """Trainer that computes the loss only for the task output tokens."""

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        input_ids = inputs["input_ids"]
        weights = inputs.get("weights", None)
        labels = inputs["labels"]

        # ic(list(inputs.keys()))
        # print()
        # breakpoint()

        # attention_mask = inputs["attention_mask"]
        # (B × T)
        input_mask = input_ids.ne(self.tokenizer.pad_token_id)
        # (B × T × 1) · (B × 1 × T) → (B × T × T)
        input_mask = input_mask.unsqueeze(-1).cpu() @ input_mask.unsqueeze(1).cpu()
        input_mask = input_mask.to(input_ids.device)
        # (B × T × T)
        attention_mask = input_mask.tril()

        bsz = input_ids.shape[0]
        seq_len = input_ids.shape[1]

        if weights is None:
            weights = torch.ones_like(input_ids)

        outputs = model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        logits = outputs["logits"]

        loss_participation_mask = weights.ne(0).logical_and(labels.gt(0))

        loc_first_tgts = weights.argmax(-1).unsqueeze(-1)
        first_k_real_tgts = torch.arange(seq_len).tile(bsz, 1).to(weights.device)
        first_k_real_tgts = first_k_real_tgts.lt(loc_first_tgts + 16)
        loss_participation_mask = loss_participation_mask.logical_and(
            first_k_real_tgts.logical_not()
        )

        flat_logits = logits[loss_participation_mask]
        flat_labels = labels[loss_participation_mask]

        loss = torch.nn.functional.cross_entropy(
            flat_logits, flat_labels, reduction="mean"
        )
        # convert nats to bits
        loss = loss / NAT_LOG_OF_2

        if loss < 0.05:
            ic(self.tokenizer.decode(input_ids[0]))
            ic(self.tokenizer.decode(input_ids[0][loss_participation_mask[0]]))
            breakpoint()

        return (loss, outputs) if return_outputs else loss

    def _get_num_items_in_batch(
        self, batch_samples: list, device: torch.device
    ) -> int | None:
        if "weights" not in batch_samples[0]:
            return sum((batch["labels"].ne(-100)).sum() for batch in batch_samples)
        return sum((batch["weights"].ne(0)).sum() for batch in batch_samples)


def tokenize(
    batch: dict, *, cfg: Config, tokenizer: AutoTokenizer, context_length: int = 1024
) -> dict:
    """Tokenize and pack sequences to minimize waste."""
    # Tokenize all texts
    all_tokens = []
    for text in batch["text"]:
        tokens = tokenizer(text, add_special_tokens=False)["input_ids"]
        all_tokens.extend(tokens)
        all_tokens.append(tokenizer.pad_token_id)  # Add separator between texts

    # Pack into fixed-length sequences
    input_batch = []
    for i in range(0, len(all_tokens) - context_length + 1, context_length):
        input_batch.append(all_tokens[i : i + context_length])

    return {"input_ids": input_batch}


def do_train(cfg: Config) -> None:
    """do_train function"""

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # load dataset from huggingface
    logger.info(f"Loading dataset: {cfg.dataset_name}")
    ds = hf_datasets.load_dataset(cfg.dataset_name)
    ds.set_format("torch")
    ic(ds)

    def collate(examples):
        obj = {
            "input_ids": torch.nn.utils.rnn.pad_sequence(
                [x["input_ids"] for x in examples],
                batch_first=True,
                padding_value=tokenizer.pad_token_id,
            ),
            "weights": torch.nn.utils.rnn.pad_sequence(
                [x["weights"] for x in examples], batch_first=True, padding_value=0
            ),
        }
        obj["labels"] = obj["input_ids"].clone()
        return obj

    collator = ReconstructionTaskCollator(tokenizer)

    # load model from huggingface
    logger.info(f"Loading model: {cfg.model_name}")
    model = AutoModelForCausalLM.from_pretrained(cfg.model_name)

    # Load the base model with specific device mapping
    model = AutoModelForCausalLM.from_pretrained(cfg.model_name, dtype=torch.bfloat16)

    if cfg.use_lora:
        if accelerator.is_main_process:
            logger.info("Loading base model for LoRA training...")

        # Configure LoRA
        lora_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.05,
            r=32,
            bias="none",
            task_type="CAUSAL_LM",
        )

        # Apply LoRA to the model
        model = get_peft_model(model, lora_config)

    else:
        if accelerator.is_main_process:
            logger.info("Loading model without LoRA...")
        model = model

    # Initialize Trainer with custom loss function if needed

    # bsz32.accum1 is 60k batches
    train_cfg = TrainingArguments(
        output_dir="./results",
        eval_strategy="steps",
        logging_strategy="steps",
        eval_steps=cfg.eval_steps,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        max_steps=cfg.max_steps,
        warmup_steps=cfg.warmup_steps,
        # num_train_epochs=1,
        gradient_accumulation_steps=cfg.accumulate_steps,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        learning_rate=cfg.learning_rate,
        bf16=True,
        push_to_hub=False,
        label_names=["labels", "weights", "attention_mask"],
        disable_tqdm=True,
    )

    trainer = TruncatedLossTrainer(
        model=model,
        tokenizer=tokenizer,
        args=train_cfg,
        data_collator=collate,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"].select(range(2000)),
    )

    # Train the model
    logger.info("Starting training...")
    trainer.train()

    # Save the model
    logger.info("Saving the model...")
    trainer.save_model("./trained_model")

    # breakpoint()
    # pass


def main() -> None:
    """main function"""
    cfg = OmegaConf.structured(Config)
    cli_cfg = OmegaConf.from_cli()
    cfg = OmegaConf.merge(cfg, cli_cfg)
    cfg = OmegaConf.to_container(cfg, resolve=True)
    try:
        cfg = Config(**cfg)
    except TypeError as e:  # pylint: disable=broad-exception-raised
        logger.error(f"Error: {e}\n\nUsage: python scratch.py")
        sys.exit(1)

    do_train(cfg)


if __name__ == "__main__":
    main()
