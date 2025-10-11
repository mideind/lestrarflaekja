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

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass
class Config:
    """Configuration for the project."""

    # dataset_name: str = "mideind/mim"
    # dataset_name: str = "mideind/mim-gold-21.05"
    dataset_name: str = "vesteinn/babylm"
    # dataset_name: str = "mideind/is_prototyping_corpus"
    model_name: str = "AI-Sweden-Models/gpt-sw3-126m"
    batch_size: int = 32
    accumulate_steps: int = 1
    warmup_steps: int = 10


class CustomLossTrainer(Trainer):
    def __init__(self, *args, loss_fn=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn = loss_fn

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        if self.loss_fn:
            loss = self.loss_fn(logits, labels)
        else:
            # Fallback to default if no custom loss is provided
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss


def tokenize(
    batch: dict, *, cfg: Config, tokenizer: AutoTokenizer, context_length: int = 1024
) -> dict:
    """Tokenize and pack sequences to minimize waste."""
    # Tokenize all texts
    all_tokens = []
    for text in batch["text"]:
        tokens = tokenizer(text, add_special_tokens=False)["input_ids"]
        all_tokens.extend(tokens)
        all_tokens.append(tokenizer.eos_token_id)  # Add separator between texts

    # Pack into fixed-length sequences
    input_batch = []
    for i in range(0, len(all_tokens) - context_length + 1, context_length):
        input_batch.append(all_tokens[i : i + context_length])

    return {"input_ids": input_batch}


def fooberino(cfg: Config) -> None:
    """fooberino function"""

    # load dataset from huggingface
    logger.info(f"Loading dataset: {cfg.dataset_name}")
    ds = hf_datasets.load_dataset(cfg.dataset_name)

    # # sample 100 datapoints from the dataset
    # ds = {
    #     "train": ds["train"].shuffle(seed=42).select(range(1000)),
    #     "valid": ds["validation"].shuffle(seed=42).select(range(100)),
    #     # "test": ds["test"].shuffle(seed=42).select(range(100)),
    # }

    ds = hf_datasets.DatasetDict(ds)

    # load model from huggingface
    logger.info(f"Loading model: {cfg.model_name}")
    model = AutoModelForCausalLM.from_pretrained(cfg.model_name)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    # fn_kwargs = {"cfg":cfg, "tokenizer":tokenizer}
    # tokenized_datasets = ds.map(
    #     # tokenize_fn, batched=True, remove_columns=ds["train"].column_names
    #     # lambda x: tokenize(cfg, x, tokenizer), batched=True, remove_columns=ds["train"].column_names
    #     tokenize, batched=True, remove_columns=ds["train"].column_names, fn_kwargs=fn_kwargs
    # )
    # data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    def collate(examples):
        return {
            "input_ids": torch.nn.utils.pad_sequence(
                [x["input_ids"] for x in examples],
                batch_first=True,
                padding_value=tokenizer.eos_token_id,
            ),
            "weights": torch.nn.utils.pad_sequence(
                [x["weights"] for x in examples], batch_first=True, padding_value=0
            ),
        }

    # train_dl = torch.utils.DataLoader(ds, batch_size=cfg.batch_size, collate_fn=collate)

    # Initialize Trainer with custom loss function if needed

    train_cfg = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        eval_strategy="steps",
        eval_steps=10,
        logging_steps=10,
        gradient_accumulation_steps=cfg.accumulate_steps,
        num_train_epochs=1,
        weight_decay=0.01,
        warmup_steps=cfg.warmup_steps,
        lr_scheduler_type="cosine",
        learning_rate=5e-4,
        save_steps=5_000,
        # fp16=False, # not allowed on mac
        fp16=True,  # not allowed on mac
        push_to_hub=False,
    )

    trainer = CustomLossTrainer(
        model=model,
        tokenizer=tokenizer,
        args=train_cfg,
        data_collator=collate,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
    )

    # Train the model
    logger.info("Starting training...")
    trainer.train()

    # Save the model
    logger.info("Saving the model...")
    trainer.save_model("./trained_model")

    breakpoint()
    pass


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

    fooberino(cfg)


if __name__ == "__main__":
    main()
