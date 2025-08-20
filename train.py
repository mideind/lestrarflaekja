"""
1. Sækja huggingface dataset

2. sækja huggingface model
    - https://huggingface.co/AI-Sweden-Models/gpt-sw3-126m/

3. Þjálfa á ["text"] lyklinum, vanilla auto-regressive transformer
- en geta breytt og fiktað í lossinum
"""

import datasets as hf_datasets
from omegaconf import OmegaConf
import logging

from dataclasses import dataclass
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
    cfg: Config, element: dict, tokenizer: AutoTokenizer, context_length: int = 1024
) -> dict:
    """Tokenize and pack sequences to minimize waste."""
    # Tokenize all texts
    all_tokens = []
    for text in element["text"]:
        tokens = tokenizer(text, add_special_tokens=False)["input_ids"]
        all_tokens.extend(tokens)
        all_tokens.append(tokenizer.eos_token_id)  # Add separator between texts
    
    # Pack into fixed-length sequences
    input_batch = []
    for i in range(0, len(all_tokens) - context_length + 1, context_length):
        input_batch.append(all_tokens[i:i + context_length])
    
    return {"input_ids": input_batch} 


def fooberino(cfg: Config) -> None:
    """fooberino function"""

    # load dataset from huggingface
    logger.info(f"Loading dataset: {cfg.dataset_name}")
    raw_dataset = hf_datasets.load_dataset(cfg.dataset_name)

    # sample 100 datapoints from the dataset
    raw_datset = {
        "train": raw_dataset["train"].shuffle(seed=42).select(range(1000)),
        "validation": raw_dataset["validation"].shuffle(seed=42).select(range(100)),
        "test": raw_dataset["test"].shuffle(seed=42).select(range(100)),
    }

    raw_dataset = hf_datasets.DatasetDict(raw_datset)

    # load model from huggingface
    logger.info(f"Loading model: {cfg.model_name}")
    model = AutoModelForCausalLM.from_pretrained(cfg.model_name)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    # tokenize the dataset
    tokenized_datasets = raw_dataset.map(
        lambda x: tokenize(cfg, x, tokenizer), batched=True, remove_columns=raw_dataset["train"].column_names
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
        fp16=False, # not allowed on mac
        push_to_hub=False
    )
    trainer = CustomLossTrainer(
        model=model,
        tokenizer=tokenizer,
        args=args,
        data_collator=data_collator,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
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
