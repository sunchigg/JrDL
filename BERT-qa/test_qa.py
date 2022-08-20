import json
import pandas as pd
import os
from argparse import ArgumentParser, Namespace

import logging
from functools import partial
from time import strftime, localtime
import datasets
from datasets import load_dataset, load_metric
from accelerate import Accelerator
import torch
from torch.utils.data.dataloader import DataLoader
import transformers
from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    default_data_collator,
    # set_seed,
)


logger = logging.getLogger(__name__)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 再改


def parse_args() -> Namespace:
    """python3.8 test_qa.py --test_file task_data/testqa_0.json --out_file ./prediction.csv --target_dir model/QA
    """
    parser = ArgumentParser()
    parser.add_argument("--test_file", type=str, default="task_data/testqa_0.json")
    parser.add_argument(
        "--target_dir", type=str,
        default="./saved/0423-2324-QA-chinese-roberta-wwm-ext")
    parser.add_argument("--test_batch_size", type=int, default=64)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--out_file", type=str, default="./prediction.csv")
    parser.add_argument("--n_best", type=int, default=20)
    parser.add_argument("--max_ans_len", type=int, default=30)

    args = parser.parse_args()
    return args


def main(args):
    args = parse_args()
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
    model = AutoModelForQuestionAnswering.from_pretrained(args.target_dir, config=config)

    # Load dataset
    raw_datasets = load_dataset("json", data_files={"test": args.test_file})
    cols = raw_datasets["test"].column_names
    args.ques_col, args.context_col, args.ans_col = "question", "context", "answer"

    test_examples = raw_datasets["test"]
    from utils_qa import prepare_pred_features
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

    model, test_dataloader = accelerator.prepare(
        model, test_dataloader
    )
    # Test
    from utils_qa import post_processing_function, create_and_fill_np_array
    logger.info("\n** Start Inferencing **")
    logger.info(f"Num test examples = {len(test_dataset)}")

    test_dataset.set_format(type="torch", columns=["attention_mask", "input_ids", "token_type_ids"])
    model.eval()
    all_start_logits = []
    all_end_logits = []
    for step, data in enumerate(test_dataloader):
        with torch.no_grad():
            outputs = model(**data)
            start_logits = outputs.start_logits
            end_logits = outputs.end_logits
            all_start_logits.append(accelerator.gather(start_logits).cpu().numpy())
            all_end_logits.append(accelerator.gather(end_logits).cpu().numpy())

    max_len = max([x.shape[1] for x in all_start_logits])
    start_logits_concat = create_and_fill_np_array(all_start_logits, test_dataset, max_len)
    end_logits_concat = create_and_fill_np_array(all_end_logits, test_dataset, max_len)
    outputs_numpy = (start_logits_concat, end_logits_concat)

    test_dataset.set_format(type=None, columns=list(test_dataset.features.keys()))
    predictions = post_processing_function(test_examples, test_dataset, outputs_numpy, args, model)
    results = {d["id"]: d["pred"].replace(",", "") for d in predictions.predictions}

    df = pd.DataFrame(columns=['id', 'answer'])
    for k, v in results.items():
        df = df.append({'id': k, 'answer': v}, ignore_index=True)
    df.to_csv(f"{args.out_file}", index=False)  # , encoding="big5"


if __name__ == "__main__":
    args = parse_args()
    main(args)
