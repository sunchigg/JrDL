import os
import json
from argparse import ArgumentParser, Namespace

import logging
from functools import partial
from time import strftime, localtime
import datasets
from datasets import load_dataset  # , load_metric
from accelerate import Accelerator
import torch
from torch.utils.data.dataloader import DataLoader
import transformers
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    default_data_collator,
)


logger = logging.getLogger(__name__)
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 再改


def parse_args() -> Namespace:
    """
    python3.8 test.py --target_dir ./saved/0510-0106_rlFrom0509-0059_512 --test_file data/public.jsonl --out_file ./submission.jsonl
    """
    parser = ArgumentParser()
    parser.add_argument("--test_file", type=str, default="data/public.jsonl")
    parser.add_argument(
        "--target_dir", type=str,
        default="./saved/0510-0106_rlFrom0509-0059_512")
    parser.add_argument("--test_batch_size", type=int, default=32)
    parser.add_argument("--out_file", type=str, default="./submission.jsonl")
    parser.add_argument(
        "--beam_size", type=int,
        default=5)  # LARGE beam_size: general, less relevent
    parser.add_argument(
        "--do_sample",
        default=False)   # False, True
    parser.add_argument(
        "--top_k", type=int,
        default=None)  # 0, 1:Greedy, 2, 5
    parser.add_argument(
        "--top_p", type=float,
        default=None)  # dynamically shrinking and expanding top-k.
    parser.add_argument(
        "--temperature", type=float,
        default=0.6)  # higher t: more uniform and more diversity

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
    model = AutoModelForSeq2SeqLM.from_pretrained(args.target_dir, config=config)

    model.resize_token_embeddings(len(tokenizer))
    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    # Load dataset
    raw_datasets = load_dataset("json", data_files={"test": args.test_file})
    cols = raw_datasets["test"].column_names
    args.id_col, args.text_col = "id", "maintext"

    from utils import prepare_pred_features
    test_examples = raw_datasets["test"]
    prepare_pred_features = partial(prepare_pred_features, args=args, tokenizer=tokenizer)
    test_dataset = test_examples.map(
        prepare_pred_features,
        batched=True,
        num_proc=4,
        remove_columns=cols,
    )

    # DataLoaders
    data_collator = default_data_collator
    test_dataloader = DataLoader(test_dataset, collate_fn=data_collator, 
                        batch_size=args.test_batch_size, num_workers=4)
    model, test_dataloader = accelerator.prepare(model, test_dataloader)

    # Test
    logger.info("\n** Start Inferencing **")
    logger.info(f"Num test examples = {len(test_dataset)}")

    model.eval()
    gen_kwargs = {
        "max_length": args.max_target_len,
        "num_beams": args.beam_size,
        "do_sample": args.do_sample,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "temperature": args.temperature,
    }
    all_preds = []
    for step, data in enumerate(test_dataloader):
        with torch.no_grad():
            generated_tokens = accelerator.unwrap_model(model).generate(
                data["input_ids"],
                attention_mask=data["attention_mask"],
                **gen_kwargs,
            )
            generated_tokens = accelerator.pad_across_processes(
                generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
            )
            generated_tokens = accelerator.gather(generated_tokens).cpu().numpy()
            decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            all_preds += decoded_preds

    with open(args.out_file, 'w') as f:
        for id_, pred in zip(test_examples[args.id_col], all_preds):
            print(json.dumps({"title": pred, "id": id_}, ensure_ascii=False), file=f)


if __name__ == "__main__":
    args = parse_args()
    main(args)
