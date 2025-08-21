# pylint: disable=unused-import,unused-argument,W0611,W0612,logging-fstring-interpolation,trailing-whitespace
# type: ignore[reportUnusedImport]
# ruff: noqa: F401

"""
1. Sækja huggingface dataset

2. Sækja huggingface model
    - https://huggingface.co/AI-Sweden-Models/gpt-sw3-126m/

3. Þjálfa á ["text"] lyklinum, vanilla auto-regressive transformer
    - en geta breytt og fiktað í lossinum

4. athuga með Garðar hvort hann hafi fínþjálfunar hástikana sem við notuðum fyrir aisweden

5. Athuga LoRA og þannig



Fyrir 1 gpu þjálfun:
```
torchrun --standalone --nnodes=1 --nproc-per-node=gpu train.py
```

fyrir 8 gpu þjálfun:
```
torchrun --standalone --nnodes=1 --nproc-per-node=8 train.py
```
"""

import functools
import logging
import sys
from typing import Mapping

import torch
import datasets as hf_datasets
from omegaconf import OmegaConf
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import PeftModel, LoraConfig, get_peft_model, prepare_model_for_kbit_training

from utils import (
    TrainConfig,
    tokenizer_fn,
)

from accelerate import Accelerator

accelerator = Accelerator()

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
        if "weights" in examples[0]:
            weights = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(example["weights"]) for example in examples],
                batch_first=True,
                padding_value=0,
            ).float()

            labels = input_ids.clone()
            labels[labels == self.tokenizer.pad_token_id] = -100

            return {"input_ids": input_ids, "labels": labels, "weights": weights}
        else:
            # TODO: Ask Haukur Barri if this is the correct way to handle labels
            labels = input_ids.clone()
            # Shift labels left by 1 position
            labels[:, :-1] = input_ids[:, 1:]
            labels[:, -1] = -100  # Ignore loss for last position

            return {
                "input_ids": input_ids,
                "labels": labels,
            }


class TruncatedLossTrainer(Trainer):
    """Trainer that computes the loss only for the task output tokens."""

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        input_ids = inputs["input_ids"]
        weights = inputs.get("weights", None)
        labels = inputs["labels"]

        if weights is None:
            weights = torch.ones_like(input_ids)

        outputs = model(input_ids=input_ids, labels=labels)
        logits = outputs["logits"]

        nonzero_weight_mask = weights.ge(0)
        # is_input_and_not_padding = labels.neq(-100) # neq() is deprecated
        is_input_and_not_padding = labels != -100
        loss_contribution_mask = nonzero_weight_mask.logical_and(
            is_input_and_not_padding
        )

        flat_logits = logits[loss_contribution_mask]
        flat_labels = labels[loss_contribution_mask]

        loss = torch.nn.functional.cross_entropy(
            flat_logits, flat_labels, reduction="mean"
        )

        ##########
        ### from super class

        # outputs = model(**inputs)
        # # Save past state if it exists
        # # TODO: this needs to be fixed and made cleaner later.
        # if self.args.past_index >= 0:
        #     self._past = outputs[self.args.past_index]

        # if labels is not None:
        #     unwrapped_model = self.accelerator.unwrap_model(model)
        #     if _is_peft_model(unwrapped_model):
        #         model_name = unwrapped_model.base_model.model._get_name()
        #     else:
        #         model_name = unwrapped_model._get_name()
        #     # User-defined compute_loss function
        #     if self.compute_loss_func is not None:
        #         loss = self.compute_loss_func(
        #             outputs, labels, num_items_in_batch=num_items_in_batch
        #         )
        #     elif model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
        #         loss = self.label_smoother(outputs, labels, shift_labels=True)
        #     else:
        #         loss = self.label_smoother(outputs, labels)

        # if (
        #     self.args.average_tokens_across_devices
        #     and (self.model_accepts_loss_kwargs or self.compute_loss_func)
        #     and num_items_in_batch is not None
        # ):
        #     loss *= self.accelerator.num_processes

        ##########

        return (loss, outputs) if return_outputs else loss

def print_trainable_parameters(model: PeftModel) -> None:
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    # print(
    #     f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    # )
    if accelerator.is_main_process:
        logger.info(
            f"trainable params: {trainable_params / 1e6:.2f}M || all params: {all_param / 1e6:.2f}M || trainable%: {100 * trainable_params / all_param:.2f}%"
        )



def fooberino(cfg: TrainConfig) -> None:
    """fooberino function"""

    # load dataset from huggingface
    if accelerator.is_main_process:
        logger.info(f"Loading dataset: {cfg.dataset_name}")
    raw_dataset = hf_datasets.load_dataset(cfg.dataset_name)

    # sample 100 datapoints from the dataset
    small_text_ds = hf_datasets.DatasetDict(
        {
            "train": raw_dataset["train"].shuffle(seed=42).select(range(1000)),
            "validation": raw_dataset["validation"].shuffle(seed=42).select(range(100)),
            "test": raw_dataset["test"].shuffle(seed=42).select(range(100)),
        }
    )

    # Load the base model with specific device mapping
    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )

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
        model = get_peft_model(base_model, lora_config)

    else:
        if accelerator.is_main_process:
            logger.info("Loading model without LoRA...")
        model = base_model
    print_trainable_parameters(model)

    model.accepts_loss_kwargs = False
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    tokenize = functools.partial(tokenizer_fn, cfg=cfg, tokenizer=tokenizer)
    # tokenize the dataset
    small_ds = small_text_ds.map(
        tokenize,
        batched=True,
        remove_columns=small_text_ds["train"].column_names,
    )

    data_collator = ReconstructionTaskCollator(tokenizer=tokenizer, mlm=False)

    # Initialize Trainer with custom loss function if needed

    args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        eval_strategy="steps",
        eval_steps=5_000,
        logging_steps=5_000,
        gradient_accumulation_steps=1,
        num_train_epochs=1,
        weight_decay=0.1,
        warmup_steps=1_000,
        lr_scheduler_type="cosine",
        learning_rate=5e-4,
        save_steps=5_000,
        push_to_hub=False,
        bf16=True,  # Enable bf16 training
    )
    trainer = TruncatedLossTrainer(
        model=model,
        tokenizer=tokenizer,
        args=args,
        data_collator=data_collator,
        train_dataset=small_ds["train"],
        eval_dataset=small_ds["validation"],
    )

    # Train the model
    if accelerator.is_main_process:
        logger.info("Starting training...")
    trainer.train()

    # Save the model
    if accelerator.is_main_process:
        logger.info("Saving the model...")
    trainer.save_model("./trained_model")


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
