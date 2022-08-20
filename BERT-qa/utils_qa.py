import collections
import logging

import numpy as np
from transformers import EvalPrediction
from tqdm.auto import tqdm
from typing import Tuple

logger = logging.getLogger(__name__)


def prepare_train_features(examples, args, tokenizer):
    pad_on_right = (tokenizer.padding_side == "right")
    max_seq_len = min(args.max_seq_len, tokenizer.model_max_length)
    tokenized_examples = tokenizer(
        examples[args.ques_col if pad_on_right else args.context_col],
        examples[args.context_col if pad_on_right else args.ques_col],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=max_seq_len,
        stride=args.stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        return_special_tokens_mask=True,
        return_token_type_ids=True,
        padding="max_length",
    )
    # https://github.com/huggingface/transformers/blob/main/examples/pytorch/question-answering/run_qa_beam_search.py
    # Label those examples!
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_examples.pop("offset_mapping")
    special_tokens = tokenized_examples.pop("special_tokens_mask")
    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []
    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        sequence_ids = tokenized_examples.sequence_ids(i)
        context_idx = 1 if pad_on_right else 0

        sample_index = sample_mapping[i]
        answer = examples[args.ans_col][sample_index]
        if len(answer) == 0:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            char_start_index = answer[0]["start"]
            char_end_index = char_start_index + len(answer[0]["text"])

            token_start_index = 0
            while sequence_ids[token_start_index] != context_idx:
                token_start_index += 1

            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != context_idx:
                token_end_index -= 1

            if not (offsets[token_start_index][0] <= char_start_index and offsets[token_end_index][1] >= char_end_index):
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= char_start_index:
                    token_start_index += 1
                token_start_index -= 1
                tokenized_examples["start_positions"].append(token_start_index)
                while offsets[token_end_index][1] >= char_end_index:
                    token_end_index -= 1
                token_end_index += 1
                tokenized_examples["end_positions"].append(token_end_index)

    return tokenized_examples


def prepare_pred_features(examples, args, tokenizer):
    pad_on_right = (tokenizer.padding_side == "right")
    max_seq_len = min(args.max_seq_len, tokenizer.model_max_length)
    tokenized_examples = tokenizer(
        examples[args.ques_col if pad_on_right else args.context_col],
        examples[args.context_col if pad_on_right else args.ques_col],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=max_seq_len,
        stride=args.stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        return_special_tokens_mask=True,
        return_token_type_ids=True,
        padding="max_length",
    )

    # Label those examples!
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    special_tokens = tokenized_examples.pop("special_tokens_mask")
    tokenized_examples["example_id"] = []
    for i, input_ids in enumerate(tokenized_examples["input_ids"]):
        cls_index = input_ids.index(tokenizer.cls_token_id)

        sequence_ids = tokenized_examples.sequence_ids(i)
        context_idx = 1 if pad_on_right else 0

        sample_index = sample_mapping[i]
        tokenized_examples["example_id"].append(examples["id"][sample_index])

        tokenized_examples["offset_mapping"][i] = [
            (o if sequence_ids[k] == context_idx else None)
            for k, o in enumerate(tokenized_examples["offset_mapping"][i])]

    return tokenized_examples


def post_processing_function(examples, features, pred_logits, args, model):
    predictions = postprocess_qa_predictions(
        examples=examples,
        features=features,
        pred_logits=pred_logits,
        n_best=args.n_best,
        max_ans_len=args.max_ans_len,
    )

    formatted_predictions = [{"id": k, "pred": v} for k, v in predictions.items()]
    references = [{"id": ex["id"], "answer": ex[args.ans_col]} for ex in examples]

    return EvalPrediction(predictions=formatted_predictions, label_ids=references)


def postprocess_qa_predictions(
    examples,
    features,
    pred_logits: Tuple[np.ndarray, np.ndarray],
    n_best: int = 20,
    max_ans_len: int = 30,
    is_world_process_zero: bool = True,
):
    # https://github.com/huggingface/transformers/blob/main/examples/pytorch/question-answering/utils_qa.py
    assert len(pred_logits) == 2, "`pred_logits` should be a tuple with two elements (start_logits, end_logits)."
    all_start_logits, all_end_logits = pred_logits

    assert pred_logits[0].shape[0] == features.shape[0], \
            f"Got {len(pred_logits[0])} pred_logits and {len(features)} features."

    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()

    logger.setLevel(logging.INFO if is_world_process_zero else logging.WARN)
    logger.info(f"Post-processing {len(examples)} example predictions split into {len(features)} features.")

    for example_index, example in enumerate(tqdm(examples)):
        feature_indices = features_per_example[example_index]

        prelim_predictions = []
        for feature_index in feature_indices:
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            offset_mapping = features[feature_index]["offset_mapping"]
            token_is_max_context = features[feature_index].get("token_is_max_context", None)

            start_indexes = np.argsort(start_logits)[-1: -n_best - 1: -1].tolist()
            end_indexes = np.argsort(end_logits)[-1: -n_best - 1: -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    if (start_index >= len(offset_mapping) or end_index >= len(offset_mapping)
                        or offset_mapping[start_index] is None or offset_mapping[end_index] is None):
                        continue
                    if end_index < start_index or end_index - start_index + 1 > max_ans_len:
                        continue
                    if token_is_max_context is not None and not token_is_max_context.get(str(start_index), False):
                        continue
                    prelim_predictions.append(
                        {
                            "offsets": (offset_mapping[start_index][0], offset_mapping[end_index][1]),
                            "score": start_logits[start_index] + end_logits[end_index],
                            "start_logit": start_logits[start_index],
                            "end_logit": end_logits[end_index],
                        }
                    )

        predictions = sorted(prelim_predictions, key=lambda x: x["score"], reverse=True)[:n_best]
        context = example["context"]
        for pred in predictions:
            offsets = pred.pop("offsets")
            pred["text"] = context[offsets[0]: offsets[1]]

        if len(predictions) == 0 or (len(predictions) == 1 and predictions[0]["text"] == ''):
            predictions.insert(0, {"text": "empty", "start_logit": 0.0, "end_logit": 0.0, "score": 0.0})

        scores = np.array([pred.pop("score") for pred in predictions])
        exp_scores = np.exp(scores - np.max(scores))
        probs = exp_scores / exp_scores.sum()
        for prob, pred in zip(probs, predictions):
            pred["probability"] = prob

        all_predictions[example["id"]] = predictions[0]["text"]
        all_nbest_json[example["id"]] = [
            {k: (float(v) if isinstance(v, (np.float16, np.float32, np.float64)) else v) for k, v in pred.items()}
            for pred in predictions
        ]

    return all_predictions


def create_and_fill_np_array(start_or_end_logits, dataset, max_len):
    """
    https://github.com/huggingface/transformers/blob/main/examples/pytorch/question-answering/run_qa_beam_search_no_trainer.py
    Create and fill numpy array of size len_of_validation_data * max_length_of_output_tensor
    Args:
        start_or_end_logits(:obj:`tensor`):
            This is the output predictions of the model. We can only enter either start or end logits.
        eval_dataset: Evaluation dataset
        max_len(:obj:`int`):
            The maximum length of the output tensor. ( See the model.eval() part for more details )
    """

    step = 0
    # create a numpy array and fill it with -100.
    logits_concat = np.full((len(dataset), max_len), -100, dtype=np.float32)
    # Now since we have create an array now we will populate it with the outputs gathered using accelerator.gather
    for i, output_logit in enumerate(start_or_end_logits):  # populate columns
        # We have to fill it such that we have to take the whole tensor and replace it on the newly created array
        # And after every iteration we have to change the step

        batch_size = output_logit.shape[0]
        cols = output_logit.shape[1]
        if step + batch_size < len(dataset):
            logits_concat[step : step + batch_size, :cols] = output_logit
        else:
            logits_concat[step:, :cols] = output_logit[: len(dataset) - step]

        step += batch_size

    return logits_concat
