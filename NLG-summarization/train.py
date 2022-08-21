import json
import os
import numpy as np
from argparse import ArgumentParser, Namespace

import logging
import math
from functools import partial
from time import strftime, localtime
import datasets
from datasets import load_dataset  # , load_metric
from accelerate import Accelerator
import torch
from torch.utils.data.dataloader import DataLoader
from torch.nn import CrossEntropyLoss
from torch.nn.utils import clip_grad_norm_
import transformers
from transformers import (
    CONFIG_MAPPING,
    # MODEL_MAPPING,
    AdamW,
    AutoConfig,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    default_data_collator,
    get_scheduler,
)


logger = logging.getLogger(__name__)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 再改


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--train_file", type=str,
        default="task_data/train_split.jsonl")
    parser.add_argument(
        "--valid_file", type=str,
        default="task_data/valid_split.jsonl")
    parser.add_argument("--max_source_len", type=int, default=512)  # 256
    parser.add_argument("--max_target_len", type=int, default=64)
    parser.add_argument("--config_name", type=str)
    parser.add_argument("--tokenizer_name", type=str)
    parser.add_argument(
        "--model_name",
        type=str,
        default="google/mt5-small",  # 再換
        )
    parser.add_argument("--train_batch_size", type=int, default=4)  # 16 OOM
    parser.add_argument("--valid_batch_size", type=int, default=4)  # 64
    parser.add_argument("--lr", type=float, default=5e-5)  # 1e-3, 2e-4
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--epoch_num", type=int, default=12)  # 10, 20
    parser.add_argument("--grad_max_norm", type=float, default=5)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--sched_type", type=str, default="linear", choices=["linear", "cosine", "constant"])
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--rl_ratio", type=float, default=0)
    parser.add_argument("--rl_top_k", type=float, default=0)
    parser.add_argument("--rl_top_p", type=float, default=0.2)
    parser.add_argument("--rl_temperature", type=float, default=0.5)
    parser.add_argument("--log_steps", type=int, default=500)  # 250
    parser.add_argument("--saved_dir", type=str, default="./saved")
    args = parser.parse_args()

    args.saved_dir = \
        os.path.join(args.saved_dir, strftime("%m%d-%H%M", localtime()))
    os.makedirs(args.saved_dir, exist_ok=True)

    return args


def compute_rl_loss(data, logits, refs, model, tokenizer, accelerator, args):
    from metric import compute_rouge
    baseline_tokens = accelerator.unwrap_model(model).generate(
        data["input_ids"],
        attention_mask=data["attention_mask"],
        max_length=args.max_target_len,
    )
    baseline_tokens = accelerator.pad_across_processes(
        baseline_tokens, dim=1, pad_index=tokenizer.pad_token_id
    )
    baseline_tokens = accelerator.gather(baseline_tokens)
    baseline_preds = tokenizer.batch_decode(baseline_tokens.cpu().numpy(), skip_special_tokens=True)

    sampled_tokens = accelerator.unwrap_model(model).generate(
        data["input_ids"],
        attention_mask=data["attention_mask"],
        max_length=args.max_target_len,
        do_sample=True,
        top_k=args.rl_top_k,
        top_p=args.rl_top_p,
        temperature=args.rl_temperature
    )
    sampled_tokens = accelerator.pad_across_processes(
        sampled_tokens, dim=1, pad_index=tokenizer.pad_token_id
    )
    sampled_tokens = accelerator.gather(sampled_tokens)
    sampled_preds = tokenizer.batch_decode(sampled_tokens.cpu().numpy(), skip_special_tokens=True)

    baseline_scores = compute_rouge(predictions=baseline_preds, references=refs, avg=False)
    baseline_rewards = \
        [(scores["rouge-1"]['f'] + scores["rouge-2"]['f'] + scores["rouge-l"]['f']) / 3 for scores in baseline_scores]
    sampled_scores = compute_rouge(predictions=sampled_preds, references=refs, avg=False)
    sampled_rewards = \
        [(scores["rouge-1"]['f'] + scores["rouge-2"]['f'] + scores["rouge-l"]['f']) / 3 for scores in sampled_scores]

    loss_fn = CrossEntropyLoss(reduction="none")
    loss_input = \
        logits[:, :sampled_tokens.shape[1], :].reshape(-1, logits.shape[-1])
    loss_target = sampled_tokens.reshape(-1)
    sampled_probs = -loss_fn(loss_input, loss_target).reshape(logits.shape[0], -1).sum(1)
    diff_rewards = \
        (torch.Tensor(baseline_rewards) - torch.Tensor(sampled_rewards)).to(sampled_probs.device)
    rl_loss = (diff_rewards * sampled_probs).mean()

    return rl_loss


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
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name, config=config)
    else:
        logger.info("train from scratch")
        model = AutoModelForSeq2SeqLM.from_config(config)

    # Resize
    model.resize_token_embeddings(len(tokenizer))
    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    # Load datasets
    from utils import prepare_train_features
    if args.valid_file:
        raw_datasets = load_dataset("json", data_files={"train": args.train_file, "valid": args.valid_file})
    else:
        raw_datasets = \
            load_dataset("json", data_files={"train": args.train_file})
    cols = raw_datasets["train"].column_names
    args.text_col, args.title_col = "maintext", "title"

    train_examples = raw_datasets["train"]
    prepare_train_features = \
        partial(prepare_train_features, args=args, tokenizer=tokenizer)
    train_dataset = train_examples.map(
        prepare_train_features,
        with_indices=True,
        batched=True,
        num_proc=4,
        remove_columns=cols,
    )

    from utils import prepare_pred_features
    if args.valid_file:
        valid_examples = raw_datasets["valid"]
        prepare_pred_features = \
            partial(prepare_pred_features, args=args, tokenizer=tokenizer)
        valid_dataset = valid_examples.map(
            prepare_pred_features,
            batched=True,
            num_proc=4,
            remove_columns=cols,
        )

    # Create DataLoaders
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=-100)
    train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=data_collator,
                            batch_size=args.train_batch_size, num_workers=4)
    if args.valid_file:
        data_collator = default_data_collator
        valid_dataloader = \
            DataLoader(valid_dataset, collate_fn=data_collator, batch_size=args.valid_batch_size, num_workers=4)
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
    optimizer = AdamW(optimizer_gparams, lr=args.lr)
    # Prepare everything with accelerator
    if args.valid_file:
        model, optimizer, train_dataloader, valid_dataloader = \
            accelerator.prepare(
                model, optimizer, train_dataloader, valid_dataloader)
    else:
        model, optimizer, train_dataloader = accelerator.prepare(
            model, optimizer, train_dataloader)
    # Scheduler
    update_steps_per_epoch = \
        math.ceil(len(train_dataloader) / args.grad_accum_steps)
    args.max_update_steps = args.epoch_num * update_steps_per_epoch
    lr_scheduler = get_scheduler(
        name=args.sched_type,
        optimizer=optimizer,
        num_warmup_steps=int(args.max_update_steps * args.warmup_ratio),
        num_training_steps=args.max_update_steps,
    )
    # Metrics
    # metrics = load_metric("")
    # Train
    total_train_batch_size = \
        args.train_batch_size * accelerator.num_processes * args.grad_accum_steps
    logger.info("\n** Start training **")
    logger.info(f"Num train examples = {len(train_dataset)}")
    logger.info(f"Num Epochs = {args.epoch_num}")
    logger.info(f"Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"Total train batch size (w/ parallel, distributed & accumulation) = {total_train_batch_size}")
    logger.info(f"Instantaneous steps per epoch = {len(train_dataloader)}")
    logger.info(f"Update steps per epoch = {update_steps_per_epoch}")
    logger.info(f"Total update steps = {args.max_update_steps}")

    from metric import compute_rouge
    max_valid_mean = 0
    for epoch in range(args.epoch_num):
        logger.info("\nEpoch {:02d} / {:02d}".format(epoch + 1, args.epoch_num))
        total_ml_loss, total_rl_loss, total_loss = 0, 0, 0
        for step, data in enumerate(train_dataloader, 1):
            model.train()
            refs = train_examples.select(data["indices"])[args.title_col]
            data = {k: v for k, v in data.items() if k != "indices"}
            outputs = model(**data)
            if args.rl_ratio < 1:
                ml_loss = outputs.loss
                total_ml_loss += ml_loss.item()
            else:
                ml_loss = 0
            if args.rl_ratio > 0:
                rl_loss = compute_rl_loss(data, outputs.logits, refs, model, tokenizer, accelerator, args)
                total_rl_loss += rl_loss.item()
            else:
                rl_loss = 0
            loss = args.rl_ratio * rl_loss + (1 - args.rl_ratio) * ml_loss
            total_loss += loss.item()
            if len(train_dataloader) % args.grad_accum_steps != 0 \
                    and len(train_dataloader) - step < args.grad_accum_steps:
                loss = loss / (len(train_dataloader) % args.grad_accum_steps)
            else:
                loss = loss / args.grad_accum_steps
            accelerator.backward(loss)

            # Update model parameters
            if step % args.grad_accum_steps == 0 or step == len(train_dataloader):
                clip_grad_norm_(model.parameters(), max_norm=args.grad_max_norm) 
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            # Log train loss
            if step % args.log_steps == 0 or step == len(train_dataloader):
                if args.rl_ratio == 0 or args.rl_ratio == 1:
                    logger.info("Train | Loss: {:.5f}".format(total_loss / step))
                else:
                    logger.info("Train | Loss: {:.5f} (ML: {:.5f}, RL:{:.5f})".format(total_loss / step, total_ml_loss / step, total_rl_loss / step))
            torch.cuda.empty_cache()
        # Evaluate
        if args.valid_file:
            model.eval()
            gen_kwargs = {
                "max_length": args.max_target_len,
            }
            all_predictions = []
            for step, data in enumerate(valid_dataloader):
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
                    all_predictions += decoded_preds

            all_references = valid_examples[args.title_col]
            valid_scores = compute_rouge(predictions=all_predictions, references=all_references)
            valid_r1 = valid_scores["rouge-1"]['f']
            valid_r2 = valid_scores["rouge-2"]['f']
            valid_rL = valid_scores["rouge-l"]['f']
            valid_mean = (valid_r1 + valid_r2 + valid_rL) / 3
            # rouge-type(baseline:f1*100)
            logger.info("Valid | rouge-1(22.0): {:.5f}, rouge-2(8.5): {:.5f}, rouge-L(20.5): {:.5f}".format(valid_r1, valid_r2, valid_rL))
            torch.cuda.empty_cache()
            if (valid_mean >= max_valid_mean) & (valid_rL > 0.18):
                # Set (valid_rL > 0.18) to prevent useless IO
                max_valid_mean = valid_mean
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
