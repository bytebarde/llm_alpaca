"""
Training code modified from https://github.com/tatsu-lab/stanford_alpaca/blob/main/train.py
"""
import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
import transformers
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorWithPadding,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)

# Constants
CACHE_DIRECTORY = "./cache/"
DEFAULT_PADDING_TOKEN = "[PAD]"
IGNORE_INDEX = -100

# Prompts Dictionary
PROMPTS = {
    "with_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "without_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="microsoft/phi-1_5")


@dataclass
class RunningArguments:
    training_mode: str = field(default="debug")


@dataclass
class TrainingArguments(TrainingArguments):
    # Basic Training Setup
    output_dir: str = field(default="./phi1.5-sft")
    num_train_epochs: int = field(default=10)
    per_device_train_batch_size: int = field(default=1)
    gradient_accumulation_steps: int = field(default=8)
    do_eval: bool = field(default=False)
    evaluation_strategy: str = field(default="epoch")
    save_strategy: str = field(default="epoch")
    save_total_limit: int = field(default=5)
    load_best_model_at_end: bool = field(default=True)
    metric_for_best_model: str = field(default="loss")
    greater_is_better: bool = field(default=False)
    label_names = ["labels"]

    # Optimizer and Scheduler
    optim: str = field(default="adamw_torch")
    learning_rate: float = field(default=2e-5)
    weight_decay: float = field(default=0.0)
    adam_epsilon: float = field(default=1e-6)
    adafactor: bool = field(default=False)
    lr_scheduler_type: str = field(default="cosine")
    warmup_ratio: float = field(default=0.03)

    # Training Process and Logging
    logging_steps: int = field(default=50)
    fp16: bool = field(default=True)


def pair_tokenize(sample, tokenizer):
    if sample["input"]:
        prompt_format = PROMPTS["with_input"]
        prompt = prompt_format.format(
            instruction=sample["instruction"], input=sample["input"]
        )
    else:
        prompt_format = PROMPTS["without_input"]
        prompt = prompt_format.format(instruction=sample["instruction"])

    # Combine prompt and output, then tokenize them
    combined_text = prompt + sample["output"]
    prompt_input_ids = tokenizer(
        prompt, truncation=True, max_length=tokenizer.model_max_length, padding=False
    )["input_ids"]
    input_ids = tokenizer(
        f"{combined_text}{tokenizer.eos_token}",
        truncation=True,
        return_tensors="pt",
        max_length=tokenizer.model_max_length,
        padding=False,
    )["input_ids"][
        0
    ]  # two d tensor with batch size 1

    prompt_length = len(prompt_input_ids)
    labels = copy.deepcopy(input_ids)
    labels[:prompt_length] = torch.full(
        (prompt_length,), IGNORE_INDEX, dtype=torch.long
    )

    return {"input_ids": input_ids, "labels": labels, "combined_text": combined_text}


def data_processing(tokenizer, mode):
    dataset_path = "vicgalle/alpaca-gpt4"
    alpaca_dataset = load_dataset(
        dataset_path, cache_dir=f"{CACHE_DIRECTORY}/datasets/"
    )["train"]
    if mode == "debug":
        alpaca_dataset = alpaca_dataset.select(range(100))
    processed_dataset = alpaca_dataset.map(
        lambda x: pair_tokenize(x, tokenizer), batched=False
    )
    processed_dataset = processed_dataset.remove_columns(
        ["instruction", "input", "output", "text", "combined_text"]
    ).with_format("torch")

    # Splitting the Dataset
    train_test_split = processed_dataset.train_test_split(test_size=0.05, shuffle=True)

    return train_test_split


# Custom Data Collator
# TODO: clean redundant params
class CustomDataCollator(DataCollatorWithPadding):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        padding=True,
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: str = "pt",
    ):
        super().__init__(
            tokenizer, padding, max_length, pad_to_multiple_of, return_tensors
        )

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        input_ids = [feature["input_ids"] for feature in features]
        labels = [feature["labels"] for feature in features]

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def initialize_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=False,
        cache_dir=f"{CACHE_DIRECTORY}/tokenizers/",
        trust_remote_code=True,
    )
    tokenizer.pad_token = DEFAULT_PADDING_TOKEN
    tokenizer.add_special_tokens({"pad_token": DEFAULT_PADDING_TOKEN})

    return tokenizer


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, TrainingArguments, RunningArguments)
    )
    model_args, training_args, running_args = parser.parse_args_into_dataclasses()
    tokenizer = initialize_tokenizer(model_args.model_name_or_path)
    train_test_split = data_processing(tokenizer, running_args.training_mode)
    data_collator = CustomDataCollator(tokenizer)
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=f"{CACHE_DIRECTORY}/models/",
        trust_remote_code=True,
    )

    model.resize_token_embeddings(len(tokenizer))
    # TODO: fixup
    training_args.fsdp_config["fsdp_transformer_layer_cls_to_wrap"] = "ParallelBlock"

    # Training Configuration
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_test_split["train"],
        eval_dataset=train_test_split["test"],
        args=training_args,
        data_collator=data_collator,
    )
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
