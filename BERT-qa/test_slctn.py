import json
import numpy as np
import os
from argparse import ArgumentParser, Namespace

import logging
from functools import partial
import datasets
from datasets import load_dataset
from accelerate import Accelerator
import torch
from torch.utils.data.dataloader import DataLoader
import transformers
from transformers import (
    AutoConfig,
    AutoModelForMultipleChoice,
    AutoTokenizer,
    default_data_collator,
)


logger = logging.getLogger(__name__)


def parse_args() -> Namespace:
    """
    python3.8 test_slctn.py --raw_test_file data/test.json --out_file ./slctn_results.json --target_dir ./model/SLC
    """
    parser = ArgumentParser()
    parser.add_argument("--raw_test_file", type=str, default="data/test.json")
    parser.add_argument("--test_file", type=str, default="task_data/test_0.json")
    parser.add_argument(
        "--target_dir", type=str,
        default="./saved/0422-1717-SLC-bert-base-chinese")
    parser.add_argument("--test_batch_size", type=int, default=32)
    parser.add_argument("--out_file", type=str, default="./slctn_results.json")

    args = parser.parse_args()
    return args


def main(args):
    with open(os.path.join(args.target_dir, "args.json"), 'r') as f:
        train_args = json.load(f)
    for k, v in train_args.items():
        if not hasattr(args, k):
            vars(args)[k] = v

    accelerator = Accelerator()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -    %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)
    # 1 process/machine.
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # Load pretrained model and tokenizer
    config = AutoConfig.from_pretrained(args.target_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.target_dir, use_fast=True)
    model = AutoModelForMultipleChoice.from_pretrained(args.target_dir, config=config)

    # Load dataset
    raw_datasets = load_dataset("json", data_files={"test": args.test_file})
    cols = raw_datasets["test"].column_names
    args.ques_col, args.para_col, args.label_col = "question", "paragraphs", "relevant"

    test_examples = raw_datasets["test"]
    from utils import prepare_pred_features
    prepare_pred_features = partial(prepare_pred_features, args=args, tokenizer=tokenizer)
    test_dataset = test_examples.map(
        prepare_pred_features,
        batched=True,
        num_proc=4,
        remove_columns=cols,
    )
    # DataLoaders
    data_collator = default_data_collator
    test_dataloader = DataLoader(test_dataset, collate_fn=data_collator, batch_size=args.test_batch_size)

    # Prepare everything with our accelerator.
    model, test_dataloader = accelerator.prepare(
        model, test_dataloader
    )
    # Test
    from utils import post_processing_function
    logger.info("\n** Start predicting **")
    logger.info(f"Num test examples = {len(test_dataset)}")

    test_dataset.set_format(columns=["attention_mask", "input_ids", "token_type_ids"])
    model.eval()
    all_logits = []
    for step, data in enumerate(test_dataloader):
        with torch.no_grad():
            outputs = model(**data)
            all_logits.append(accelerator.gather(outputs.logits).squeeze(-1).cpu().numpy())
    outputs_numpy = np.concatenate(all_logits, axis=0)

    test_dataset.set_format(columns=list(test_dataset.features.keys()))
    predictions = post_processing_function(test_examples, test_dataset, outputs_numpy, args)
    with open(args.raw_test_file, 'r') as f:
        raw_test_data = json.load(f)
    example_id_to_index = {d["id"]: i for i, d in enumerate(raw_test_data)}
    results = []
    for i, d in enumerate(predictions.predictions):
        index = example_id_to_index[d["id"]]
        results.append({**raw_test_data[index], "relevant": raw_test_data[index]["paragraphs"][d["pred"]]})

    os.makedirs(os.path.dirname(args.out_file), exist_ok=True)
    with open(args.out_file, 'w') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    args = parse_args()
    main(args)
