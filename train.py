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
    # dataset_name: str = "vesteinn/babylm"
    dataset_name: str = "mideind/lestur.soup.igc"
    # dataset_name: str = "mideind/is_prototyping_corpus"
    # model_name: str = "AI-Sweden-Models/gpt-sw3-126m"
    # model_name: str = "AI-Sweden-Models/gpt-sw3-356m"
    # model_name: str = "AI-Sweden-Models/gpt-sw3-1.3b"
    model_name: str = "mideind/byt5-large-spanmask-pretrain"
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

    # weight_decay=0.01,
    # lr_scheduler_type="cosine",


# class ReconstructionTaskCollator(DataCollatorForLanguageModeling):
#     """Collator for the reconstruction task."""

#     def __init__(self, tokenizer):
#         super().__init__(tokenizer, mlm=False)

#     def torch_call(self, examples: list[dict]) -> dict:
#         # the super method does not handle our dict keys

#         assert "weights" in examples[0]

#         # Handle dict or lists with proper padding and conversion to tensor.

#         if self.seed and self.generator is None:
#             # If we have a seed, we need to create a generator object. Subsequent calls to this function will use the same generator.
#             # If no seed supplied, we will use the global RNG
#             self.create_rng()

#         bsz = len(examples)

#         input_ids = torch.nn.utils.rnn.pad_sequence(
#             # [torch.tensor(example["input_ids"]) for example in examples],
#             [example["input_ids"] for example in examples],
#             batch_first=True,
#             padding_value=self.tokenizer.pad_token_id,
#         )

#         weights = torch.nn.utils.rnn.pad_sequence(
#             # [torch.tensor(example["weights"]) for example in examples],
#             [example["weights"] for example in examples],
#             batch_first=True,
#             padding_value=0,
#         ).float()

#         # (B × T)
#         decoder_attn_mask = input_ids.ne(self.tokenizer.pad_token_id)
#         # (B × T × 1) · (B × 1 × T) → (B × T × T)
#         decoder_attn_mask = decoder_attn_mask.unsqueeze(-1) @ decoder_attn_mask.unsqueeze(1)
#         # (B × T × T)
#         attention_mask = decoder_attn_mask.tril()

#         # shift the input so we predict the next token
#         labels = input_ids.roll(-1)
#         labels[:, -1] = self.tokenizer.pad_token_id
#         labels[labels.eq(self.tokenizer.pad_token_id)] = -100

#         return {
#             "input_ids": input_ids,
#             "labels": labels,
#             "weights": weights,
#             "attention_mask": attention_mask,
#         }


class TruncatedLossTrainer(Trainer):
    """Trainer that computes the loss only for the task output tokens."""

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        # decoder-only model
        if "input_ids" in inputs:
            input_ids = inputs["input_ids"]
            labels = inputs["labels"]
            attention_mask = inputs["attention_mask"]
            mask_keep_loss = inputs["mask_keep_loss"]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs["logits"]

            # filter out padding and loss-truncation
            flat_logits = logits[mask_keep_loss]
            flat_labels = labels[mask_keep_loss]

            # division by ln(2) converts nats to bits
            loss = (
                torch.nn.functional.cross_entropy(
                    flat_logits, flat_labels, reduction="mean"
                )
                / NAT_LOG_OF_2
            )

            return (loss, outputs) if return_outputs else loss

        else:
            # encoder decoder model (byt5)
            assert "enc_input_ids" in inputs
            enc_input_ids = inputs["enc_input_ids"]
            unshifted_labels = inputs["unshifted_labels"]

            # we assume the t5 class shifts the labels, converts -100 to padding and constructs attention mask

            outputs = model(enc_input_ids, labels=unshifted_labels)

    def _get_num_items_in_batch(
        self, batch_samples: list, device: torch.device
    ) -> int | None:
        # decoder-only model
        if "weights" in batch_samples[0]:
            return sum((batch["weights"].gt(0)).sum() for batch in batch_samples)
        elif "labels" in batch_samples[0]:
            assert "labels" in batch_samples[0]
            return sum((batch["labels"].ne(-100)).sum() for batch in batch_samples)
        else:
            assert "unshifted_labels" in batch_samples[0]
            return sum(
                (batch["unshifted_labels"].ne(-100)).sum() for batch in batch_samples
            )


def do_train(cfg: Config) -> None:
    """do_train function"""

    # tokenizer = AutoTokenizer.from_pretrained("AI-Sweden-Models/gpt-sw3-356m")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    byte_tokenizer = AutoTokenizer.from_pretrained("google/byt5-small")

    # load dataset from huggingface
    logger.info(f"Loading dataset: {cfg.dataset_name}")
    ds = hf_datasets.load_dataset(cfg.dataset_name)
    ds.set_format("torch")

    def convert_example_to_byt5(example, tokenizer):
        input_ids = example["input_ids"]
        weights = example["weights"]

        source_ids = input_ids[weights.eq(0)]
        target_ids = input_ids[weights.gt(0)]

        enc_input_ids = torch.tensor(tokenizer.decode(source_ids).encode("utf8") + 3)
        dec_input_ids = torch.tensor(tokenizer.decode(target_ids).encode("utf8") + 3)
        weights = torch.ones_like(dec_input_ids).float()

        return {
            "enc_input_ids": enc_input_ids,
            "dec_input_ids": dec_input_ids,
        }

    def collate_for_byt5(examples):
        examples = [convert_example_to_byt5(ex) for ex in examples]

        enc_input_ids = torch.nn.utils.rnn.pad_sequence(
            [ex["enc_input_ids"] for ex in examples],
            batch_first=True,
            padding_value=tokenizer.pad_token_id,
        )

        dec_input_ids = torch.nn.utils.rnn.pad_sequence(
            [ex["dec_input_ids"] for ex in examples],
            batch_first=True,
            padding_value=-100,
        )

        return {
            "enc_input_ids": enc_input_ids,
            "unshifted_labels": dec_input_ids,
        }

    def collate(examples):
        bsz = len(examples)
        pad_token_id = tokenizer.pad_token_id

        input_ids = torch.nn.utils.rnn.pad_sequence(
            [ex["input_ids"] for ex in examples],
            batch_first=True,
            padding_value=tokenizer.pad_token_id,
        )

        weights = torch.nn.utils.rnn.pad_sequence(
            [ex["weights"] for ex in examples],
            batch_first=True,
            padding_value=0,
        ).float()

        # input_ids = torch.nn.utils.rnn.pad_sequence(
        #     [ex["input_ids"][ex["weights"].gt(0)] for ex in examples],
        #     batch_first=True,
        #     padding_value=self.tokenizer.pad_token_id,
        # )
        # weights = torch.nn.utils.rnn.pad_sequence(
        #     [ex["weights"][ex["weights"].gt(0)] for ex in examples],
        #     batch_first=True,
        #     padding_value=0,
        # ).float()

        # (B × T)
        decoder_attn_mask = input_ids.ne(pad_token_id)
        # (B × T × 1) ⨀ (B × 1 × T) → (B × T × T)
        decoder_attn_mask = (
            decoder_attn_mask[:, :, None] * decoder_attn_mask[:, None, :]
        )
        # (B × T × T)
        attention_mask = decoder_attn_mask.tril()

        # shift the input so we predict the next token
        labels = input_ids.roll(-1)
        labels[:, -1] = pad_token_id
        labels[labels.eq(pad_token_id)] = -100

        mask_keep_loss = weights.gt(0).logical_and(labels.gt(0))

        # seq_len = input_ids.shape[1]
        # # ignore first k tokens of real targets
        # loc_first_tgts = weights.argmax(-1).unsqueeze(-1)
        # first_k_real_tgts = torch.arange(seq_len).tile(bsz, 1).to(weights.device)
        # first_k_real_tgts = first_k_real_tgts.lt(loc_first_tgts + 16)
        # mask_keep_loss = mask_keep_loss.logical_and(
        #     first_k_real_tgts.logical_not()
        # )

        return {
            "input_ids": input_ids,
            "labels": labels,
            "weights": weights,
            "attention_mask": attention_mask,
            "mask_keep_loss": mask_keep_loss,
        }

    # load model from huggingface
    logger.info(f"Loading model: {cfg.model_name}")

    if "byt5" in cfg.model_name:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            cfg.model_name, dtype=torch.bfloat16
        )
    else:
        # Load the base model with specific device mapping
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name, dtype=torch.bfloat16
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
        per_device_eval_batch_size=cfg.batch_size * 2,
        weight_decay=0.001,
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
        data_collator=collate_for_byt5 if "t5" in cfg.model_name else collate,
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
