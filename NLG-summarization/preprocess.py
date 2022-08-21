from argparse import ArgumentParser, Namespace
import numpy as np
import os


def parse_args() -> Namespace:
    """for training
    python3.8 preprocess.py -a ./data/train.jsonl -o ./task_data
    """
    parser = ArgumentParser()
    parser.add_argument(
        "-a",
        "--article_data",
        type=str,
        default="./data/train.jsonl",
    )
    parser.add_argument(
        "-o",
        "--out_dir",
        type=str,
        default="./task_data",
    )
    parser.add_argument(
        "-s",
        "--split_ratio",
        type=float,
        default=0.2,  # 0.16, 0.2
    )

    args = parser.parse_args()
    return args


def main(args):
    np.random.seed(13)
    os.makedirs(args.out_dir, exist_ok=True)
    with open(args.article_data, 'r') as f:
        data = f.readlines()

    all_indices = np.random.permutation(np.arange(len(data)))
    cut = int(len(data) * args.split_ratio)
    with open(f"{args.out_dir}/train_split.jsonl", 'w') as f:
        for i in all_indices[cut:].tolist():
            f.write(data[i])
    with open(f"{args.out_dir}/valid_split.jsonl", 'w') as f:
        for i in all_indices[:cut].tolist():
            f.write(data[i])


if __name__ == "__main__":
    args = parse_args()
    main(args)
