import logging


logger = logging.getLogger(__name__)


def prepare_train_features(examples, indices, args, tokenizer):
    """
    https://github.com/huggingface/transformers/issues/14531
    """
    inputs = examples[args.text_col]
    targets = examples[args.title_col]

    model_inputs = tokenizer(inputs, max_length=args.max_source_len, padding="max_length", truncation=True)

    # Tokenizer
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=args.max_target_len, padding="max_length", truncation=True)

    # Replace all tokenizer.pad_token_id in the labels by -100 as we want to ignore padding in the loss.
    labels["input_ids"] = [[(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]]

    model_inputs["labels"] = labels["input_ids"]
    model_inputs["indices"] = indices

    return model_inputs


def prepare_pred_features(examples, args, tokenizer):
    inputs = examples[args.text_col]
    model_inputs = tokenizer(inputs, max_length=args.max_source_len, padding="max_length", truncation=True)
    return model_inputs
