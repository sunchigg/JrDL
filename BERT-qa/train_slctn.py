import json
import os
import numpy as np
from argparse import ArgumentParser, Namespace

import logging
import math
from functools import partial
from time import strftime, localtime
import datasets
from datasets import load_dataset, load_metric
from accelerate import Accelerator
import torch
from torch.utils.data.dataloader import DataLoader
import transformers
from transformers import (
    CONFIG_MAPPING,
    AdamW,
    AutoConfig,
    AutoModelForMultipleChoice,
    AutoTokenizer,
    default_data_collator,
    get_scheduler,
)


logger = logging.getLogger(__name__)
os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # 再改


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--train_file", type=str, default="task_data/train_0.json")
    parser.add_argument("--valid_file", type=str, default="task_data/valid_0.json")
    parser.add_argument("--neg_num", type=int, default=4)
    parser.add_argument("--max_seq_len", type=int, default=512)  # 512 better but CUDA out of memory.
    parser.add_argument("--stride", type=int, default=64)  # 128
    parser.add_argument("--config_name", type=str)
    parser.add_argument("--tokenizer_name", type=str)
    parser.add_argument(
        "--model_name",
        type=str,
        default="bert-base-chinese",  # 再換
    )  # hfl/chinese-xlnet-base, hfl/chinese-bert-wwm-ext, bert-base-chinese
    parser.add_argument("--train_batch_size", type=int, default=2)  # 8
    parser.add_argument("--valid_batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)  # 3e-5
    parser.add_argument("--weight_decay", type=float, default=0.01)  # 0.01
    parser.add_argument("--epoch_num", type=int, default=5)  # 3
    parser.add_argument("--grad_accum_steps", type=int, default=16)
    parser.add_argument("--sched_type", type=str, default="linear", choices=["linear", "cosine", "constant"])
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--log_steps", type=int, default=500)
    parser.add_argument("--eval_steps", type=int, default=2000)
    parser.add_argument("--saved_dir", type=str, default="./saved")
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    args.saved_dir = os.path.join(args.saved_dir, strftime("%m%d-%H%M", localtime()))
    os.makedirs(args.saved_dir, exist_ok=True)

    return args


def main(args):
    args = parse_args()
    logger.info("Save args to {}/".format(os.path.join(args.saved_dir, "args.json")))
    with open(os.path.join(args.saved_dir, "args.json"), 'w') as f:
        json.dump(vars(args), f, indent=4)

    accelerator = Accelerator()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -    %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.addHandler(logging.FileHandler(os.path.join(args.saved_dir, "log")))
    logger.info(accelerator.state)
    # 1 process/machine
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
    # Load pretrained model and tokenizer
    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name)
    elif args.model_name:
        config = AutoConfig.from_pretrained(args.model_name)
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True, do_lower_case=True)
    elif args.model_name:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True, do_lower_case=True)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    logger.info("Save tokenizer to {}/".format(os.path.join(args.saved_dir, "tokenizer")))
    tokenizer.save_pretrained(args.saved_dir)

    if args.model_name:
        model = AutoModelForMultipleChoice.from_pretrained(args.model_name, config=config)
    else:
        logger.info("train from scratch")
        model = AutoModelForMultipleChoice.from_config(config)

    # Load datasets
    from utils import prepare_train_features
    if args.valid_file:
        raw_datasets = load_dataset("json", data_files={"train": args.train_file, "valid": args.valid_file})
    else:
        raw_datasets = load_dataset("json", data_files={"train": args.train_file})
    cols = raw_datasets["train"].column_names
    args.ques_col, args.para_col, args.label_col = "question", "paragraphs", "relevant"

    train_examples = raw_datasets["train"]
    prepare_train_features = partial(prepare_train_features, args=args, tokenizer=tokenizer)
    train_dataset = train_examples.map(
        prepare_train_features,
        batched=True,
        num_proc=4,
        remove_columns=cols,
    )

    from utils import prepare_pred_features
    if args.valid_file:
        valid_examples = raw_datasets["valid"]
        prepare_pred_features = partial(prepare_pred_features, args=args, tokenizer=tokenizer)
        valid_dataset = valid_examples.map(
            prepare_pred_features,
            batched=True,
            num_proc=4,
            remove_columns=cols,
        )
    # Create DataLoaders
    from utils import data_collator_with_neg_sampling
    data_collator = partial(data_collator_with_neg_sampling, args=args)
    train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=data_collator, 
                            batch_size=args.train_batch_size, num_workers=4)
    if args.valid_file:
        data_collator = default_data_collator
        valid_dataloader = DataLoader(valid_dataset, collate_fn=data_collator, 
                            batch_size=args.valid_batch_size, num_workers=4) 
    # Optimizer
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_gparams = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_gparams, lr=args.lr, betas=(0.9, 0.98))
    # Prepare everything with accelerator.
    if args.valid_file:
        model, optimizer, train_dataloader, valid_dataloader = accelerator.prepare(
            model, optimizer, train_dataloader, valid_dataloader
        )
    else:
        model, optimizer, train_dataloader = accelerator.prepare(
            model, optimizer, train_dataloader
        )
    # Scheduler
    update_steps_per_epoch = math.ceil(len(train_dataloader) / args.grad_accum_steps)
    args.max_update_steps = args.epoch_num * update_steps_per_epoch
    lr_scheduler = get_scheduler(
        name=args.sched_type,
        optimizer=optimizer,
        num_warmup_steps=int(args.max_update_steps * args.warmup_ratio),
        num_training_steps=args.max_update_steps,
    )
    # Metrics
    metrics = load_metric("./metric_slctn.py")
    # Train
    total_train_batch_size = args.train_batch_size * accelerator.num_processes * args.grad_accum_steps
    logger.info("\n** Start training **")
    logger.info(f"Num train examples = {len(train_dataset)}")
    logger.info(f"Num Epochs = {args.epoch_num}")
    logger.info(f"Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"Total train batch size (w/ parallel, distributed & accumulation) = {total_train_batch_size}")
    logger.info(f"Instantaneous steps per epoch = {len(train_dataloader)}")
    logger.info(f"Update steps per epoch = {update_steps_per_epoch}")
    logger.info(f"Total update steps = {args.max_update_steps}")

    max_valid_acc = 0
    for epoch in range(args.epoch_num):
        logger.info("\nEpoch {:02d} / {:02d}".format(epoch + 1, args.epoch_num))
        total_loss = 0
        for step, data in enumerate(train_dataloader, 1):
            model.train()
            outputs = model(**data)
            loss = outputs.loss
            total_loss += loss.item()
            if len(train_dataloader) % args.grad_accum_steps != 0 \
                    and len(train_dataloader) - step < args.grad_accum_steps:
                loss = loss / (len(train_dataloader) % args.grad_accum_steps)
            else:
                loss = loss / args.grad_accum_steps
            accelerator.backward(loss)

            # Update model parameters
            if step % args.grad_accum_steps == 0 or step == len(train_dataloader):
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            # Log train loss
            if step % args.log_steps == 0 or step == len(train_dataloader):
                logger.info("Train | Loss: {:.5f}".format(total_loss / step))
        # Evaluate
        from utils import post_processing_function, postprocess_relchoice_predictions
        if args.valid_file and (step % args.eval_steps == 0 or step == len(train_dataloader)):
            valid_dataset.set_format(columns=["attention_mask", "input_ids", "token_type_ids"])
            model.eval()
            all_logits = []
            for step, data in enumerate(valid_dataloader):
                with torch.no_grad():
                    outputs = model(**data)
                    all_logits.append(accelerator.gather(outputs.logits).squeeze(-1).cpu().numpy())

            outputs_numpy = np.concatenate(all_logits, axis=0)

            valid_dataset.set_format(columns=list(valid_dataset.features.keys()))
            predictions = post_processing_function(valid_examples, valid_dataset, outputs_numpy, args)
            valid_acc = metrics.compute(predictions=predictions.predictions, references=predictions.label_ids)
            logger.info("Valid | Acc: {:.5f}".format(valid_acc))
            if valid_acc >= max_valid_acc:
                max_valid_acc = valid_acc
                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(args.saved_dir, save_function=accelerator.save)
                logger.info("Save config and model to {}/".format(args.saved_dir))

    if not args.valid_file:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(args.saved_dir, save_function=accelerator.save)
        logger.info("Save config and model to {}/".format(args.saved_dir))


if __name__ == "__main__":
    args = parse_args()
    main(args)
