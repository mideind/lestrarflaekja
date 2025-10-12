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
    use_lora: bool = False


class ReconstructionTaskCollator(DataCollatorForLanguageModeling):
    """Collator for the reconstruction task."""

    def __init__(self, tokenizer):
        super().__init__(tokenizer, mlm=False)

    def torch_call(self, examples: list[dict]) -> dict:
        # the super method does not handle our dict keys

        ic(examples[0].keys())
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

        if "weights" in examples[0]:
            weights = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(example["weights"]) for example in examples],
                batch_first=True,
                padding_value=0,
            ).float()

            labels = input_ids.clone()
            labels[labels == self.tokenizer.pad_token_id] = -100

            return {"input_ids": input_ids, "labels": labels, "weights": weights}

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
        #         model_name = unwrapped_model.model.model._get_name()
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


def do_train(cfg: Config) -> None:
    """do_train function"""

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    # load dataset from huggingface
    logger.info(f"Loading dataset: {cfg.dataset_name}")
    ds = hf_datasets.load_dataset(cfg.dataset_name)
    ds.set_format("torch")
    ic(ds)

    def collate(examples):
        return {
            "input_ids": torch.nn.utils.rnn.pad_sequence(
                [x["input_ids"] for x in examples],
                batch_first=True,
                padding_value=tokenizer.eos_token_id,
            ),
            "weights": torch.nn.utils.rnn.pad_sequence(
                [x["weights"] for x in examples], batch_first=True, padding_value=0
            ),
        }

    collator = ReconstructionTaskCollator(tokenizer)

    # # sample 100 datapoints from the dataset
    # ds = {
    #     "train": ds["train"].shuffle(seed=42).select(range(1000)),
    #     "valid": ds["validation"].shuffle(seed=42).select(range(100)),
    #     # "test": ds["test"].shuffle(seed=42).select(range(100)),
    # }

    # fn_kwargs = {"cfg":cfg, "tokenizer":tokenizer}
    # tokenized_datasets = ds.map(
    #     # tokenize_fn, batched=True, remove_columns=ds["train"].column_names
    #     # lambda x: tokenize(cfg, x, tokenizer), batched=True, remove_columns=ds["train"].column_names
    #     tokenize, batched=True, remove_columns=ds["train"].column_names, fn_kwargs=fn_kwargs
    # )
    # data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    # train_dl = torch.utils.DataLoader(ds, batch_size=cfg.batch_size, collate_fn=collate)

    # load model from huggingface
    logger.info(f"Loading model: {cfg.model_name}")
    model = AutoModelForCausalLM.from_pretrained(cfg.model_name)

    # Load the base model with specific device mapping
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name, torch_dtype=torch.bfloat16
    )
    # model.accepts_loss_kwargs = False

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
        bf16=True,  # not allowed on mac
        push_to_hub=False,
        label_names=["labels", "weights"],
    )

    trainer = TruncatedLossTrainer(
        model=model,
        tokenizer=tokenizer,
        args=train_cfg,
        data_collator=collate,
        # data_collator=collator,
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

    do_train(cfg)


if __name__ == "__main__":
    main()
