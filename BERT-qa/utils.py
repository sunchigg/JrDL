import random
import torch
import collections
import logging

import numpy as np
from transformers import EvalPrediction
from tqdm.auto import tqdm


logger = logging.getLogger(__name__)


def prepare_train_features(examples, args, tokenizer):
    pad_on_right = (tokenizer.padding_side == "right")
    first_sentences, second_sentences, sentence_counts, labels = [], [], [], []
    for ques, paras, label in zip(examples[args.ques_col], examples[args.para_col], examples[args.label_col]):
        if len(paras) < args.neg_num + 1:
            paras += ['' for i in range(args.neg_num + 1 - len(paras))]
        sentence_counts.append(len(paras))
        labels.append(0)    # For easier negative sampling implementation in collator.
        for i in [label] + list(range(0, label)) + list(range(label + 1, len(paras))):
            if pad_on_right:
                first_sentences.append(ques)
                second_sentences.append(paras[i])
            else:
                first_sentences.append(paras[i])
                second_sentences.append(ques)

    max_seq_len = min(args.max_seq_len, tokenizer.model_max_length)
    tokenized_inputs = tokenizer(
        first_sentences,
        second_sentences,
        truncation="only_second" if pad_on_right else "only_first",
        max_length=max_seq_len,
        padding="max_length",
    )

    tokenized_examples = dict()
    for k, v in tokenized_inputs.items():
        tokenized_examples[k] = []
        curr = 0
        for c in sentence_counts:
            tokenized_examples[k].append(v[curr: curr + c])
            curr += c
        assert curr == len(v)
    tokenized_examples["labels"] = labels

    return tokenized_examples


def prepare_pred_features(examples, args, tokenizer):
    pad_on_right = (tokenizer.padding_side == "right")
    first_sentences, second_sentences, sample_indices = [], [], []
    for i, (ques, paras) in enumerate(zip(examples[args.ques_col], examples[args.para_col])):
        for p in paras:
            if pad_on_right:
                first_sentences.append(ques)
                second_sentences.append(p)
            else:
                first_sentences.append(p)
                second_sentences.append(ques)
            sample_indices.append(i)

    max_seq_len = min(args.max_seq_len, tokenizer.model_max_length)
    tokenized_inputs = tokenizer(
        first_sentences,
        second_sentences,
        truncation="only_second" if pad_on_right else "only_first",
        max_length=max_seq_len,
        padding="max_length",
    )

    tokenized_examples = dict()
    for k, v in tokenized_inputs.items():
        tokenized_examples[k] = [[v_] for v_ in v]
    tokenized_examples["example_id"] = [examples["id"][sample_index] for sample_index in sample_indices]

    return tokenized_examples


def data_collator_with_neg_sampling(features, args):
    first = features[0]
    batch = {}

    all_para_counts = [len(f["input_ids"]) for f in features]
    select_indices = [[0] + sorted(random.sample(range(1, para_count), k=args.neg_num)) \
                        for para_count in all_para_counts]

    batch["labels"] = torch.tensor([f["labels"] for f in features], dtype=torch.long)

    for k, v in first.items():
        if k != "labels" and not isinstance(v, str):
            batch[k] = torch.stack([torch.tensor(f[k])[i] for f, i in zip(features, select_indices)])

    return batch


def postprocess_relchoice_predictions(
    examples,
    features,
    pred_logits,
    is_world_process_zero: bool = True,
):
    assert pred_logits.shape[0] == features.shape[0], \
            f"Got {len(pred_logits[0])} pred_logits and {len(features)} features."

    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)

    all_predictions = collections.OrderedDict()

    logger.setLevel(logging.INFO if is_world_process_zero else logging.WARN)
    logger.info(f"Post-processing {len(examples)} example predictions split into {len(features)} features.")

    for example_index, example in enumerate(tqdm(examples)):
        feature_indices = features_per_example[example_index]

        prelim_predictions = []
        for feature_index in feature_indices:
            prelim_predictions.append(pred_logits[feature_index])
        prediction = np.argmax(prelim_predictions)

        all_predictions[example["id"]] = prediction

    return all_predictions


def post_processing_function(examples, features, pred_logits, args):
    predictions = postprocess_relchoice_predictions(
        examples=examples,
        features=features,
        pred_logits=pred_logits,
    )

    formatted_predictions = [{"id": k, "pred": v} for k, v in predictions.items()]
    references = [{"id": ex["id"], "label": ex[args.label_col]} for ex in examples]

    return EvalPrediction(predictions=formatted_predictions, label_ids=references)
