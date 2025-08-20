# pylint: disable=unused-import,unused-argument,W0611,logging-fstring-interpolation,trailing-whitespace
# type: ignore[reportUnusedImport]
# ruff: noqa: F401

"""
1. Sækja huggingface dataset

2. Sækja huggingface model
    - https://huggingface.co/AI-Sweden-Models/gpt-sw3-126m/

3. Þjálfa á ["text"] lyklinum, vanilla auto-regressive transformer
    - en geta breytt og fiktað í lossinum

4. athuga með Garðar hvort hann hafi fínþjálfunar hástikana sem við notuðum fyrir aisweden
"""

import functools
import logging
import sys

import datasets as hf_datasets
from omegaconf import OmegaConf
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from utils import (
    TrainConfig,
    tokenizer_fn,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


class ReconstructionTaskCollator(DataCollatorForLanguageModeling):
    """Collator for the reconstruction task."""

    def torch_call(self, examples: list[dict]) -> dict:
        # the super method does not handle our dict keys
        # TODO: WIP
        breakpoint()


class CustomTrainer(Trainer):
    def __init__(self, *args, loss_fn=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn = loss_fn

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs["labels"]
        outputs = model(**inputs)
        logits = outputs["logits"]

        if self.loss_fn:
            loss = self.loss_fn(logits, labels)
        else:
            # Fallback to default if no custom loss is provided
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss


def fooberino(cfg: TrainConfig) -> None:
    """fooberino function"""

    # load dataset from huggingface
    logger.info(f"Loading dataset: {cfg.dataset_name}")
    raw_dataset = hf_datasets.load_dataset(cfg.dataset_name)

    # sample 100 datapoints from the dataset
    small_text_ds = hf_datasets.DatasetDict({
        "train": raw_dataset["train"].shuffle(seed=42).select(range(1000)),
        "validation": raw_dataset["validation"].shuffle(seed=42).select(range(100)),
        "test": raw_dataset["test"].shuffle(seed=42).select(range(100)),
    })

    # load model from huggingface
    logger.info(f"Loading model: {cfg.model_name}")
    model = AutoModelForCausalLM.from_pretrained(cfg.model_name)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    tokenize = functools.partial(tokenizer_fn, cfg=cfg, tokenizer=tokenizer)
    # tokenize the dataset
    small_ds = small_text_ds.map(
        tokenize,
        batched=True,
        remove_columns=small_text_ds["train"].column_names,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Initialize Trainer with custom loss function if needed

    args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        eval_strategy="steps",
        eval_steps=5_000,
        logging_steps=5_000,
        gradient_accumulation_steps=8,
        num_train_epochs=1,
        weight_decay=0.1,
        warmup_steps=1_000,
        lr_scheduler_type="cosine",
        learning_rate=5e-4,
        save_steps=5_000,
        fp16=False,  # not allowed on mac
        push_to_hub=False,
    )
    trainer = CustomTrainer(
        model=model,
        tokenizer=tokenizer,
        args=args,
        data_collator=data_collator,
        train_dataset=small_ds["train"],
        eval_dataset=small_ds["validation"],
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
    cfg = OmegaConf.structured(TrainConfig)
    cli_cfg = OmegaConf.from_cli()
    cfg = OmegaConf.merge(cfg, cli_cfg)
    cfg = OmegaConf.to_container(cfg, resolve=True)
    try:
        cfg = TrainConfig(**cfg)
    except TypeError as e:  # pylint: disable=broad-exception-raised
        logger.error(f"Error: {e}\n\nUsage: python scratch.py")
        sys.exit(1)

    fooberino(cfg)


if __name__ == "__main__":
    main()
