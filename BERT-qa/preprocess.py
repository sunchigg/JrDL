import os
import json
from argparse import ArgumentParser, Namespace
import numpy as np


def parse_args() -> Namespace:
    """for test
    python preprocess.py -q data/test.json -c data/context.json -o test -s 0
    python preprocess.py -t qa -o trainqa -s 0
    """
    parser = ArgumentParser()
    parser.add_argument(
        "-q",
        "--question_data",
        type=str,
        default="./data/train.json",  # ./data/valid.json, ./data/test.json
    )
    parser.add_argument(
        "-c",
        "--context_data",
        type=str,
        default="./data/context.json",
    )
    parser.add_argument(
        "-o",
        "--out_prefix",
        type=str,
        default="train",  # train, valid, test, trainqa, validqa, testqa
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./task_data",
    )
    parser.add_argument(
        "-t",
        "--task_slctn_or_qa",
        type=str,
        default="slctn"  # qa
    )
    parser.add_argument(
        "-s",
        "--split_ratio",
        type=float,
        default=0,  # 0.16
    )

    args = parser.parse_args()
    return args


def main(args):
    np.random.seed(13)
    os.makedirs(args.out_dir, exist_ok=True)

    # load rawdata
    with open(args.question_data, 'r') as f:
        question_data = json.load(f)
    with open(args.context_data, 'r') as f:
        context_data = json.load(f)

    if args.split_ratio > 0:
        all_indices = np.random.permutation(np.arange(len(question_data)))
        cut = int(len(question_data) * args.split_ratio)
        split_indices = [all_indices[cut:].tolist(), all_indices[:cut].tolist()]
    else:
        split_indices = [np.arange(len(question_data)).tolist()]

    for si, ids in enumerate(split_indices):
        with open(os.path.join(args.out_dir, args.out_prefix + "_{}.json".format(si)), 'w') as f:
            if args.task_slctn_or_qa == 'slctn':
                for i in ids:
                    q_data = question_data[i]
                    paragraphs = []
                    rel = None
                    for pi, p in enumerate(q_data["paragraphs"]):
                        paragraphs.append(context_data[p])
                        if p == q_data.get("relevant", None):
                            rel = pi
                    data = {"id": q_data["id"],
                            "question": q_data["question"],
                            "paragraphs": paragraphs,
                            "relevant": rel}
                    print(json.dumps(data, ensure_ascii=False), file=f)
            if args.task_slctn_or_qa == 'qa':
                for i in ids:
                    q_data = question_data[i]
                    if "answer" in q_data:
                        res = [q_data.get("answer")]
                    else:
                        res = []
                    data = {"id": q_data["id"],
                            "question": q_data["question"],
                            "context": context_data[q_data["relevant"]],
                            "answer": res}
                    print(json.dumps(data, ensure_ascii=False), file=f)


if __name__ == "__main__":
    args = parse_args()
    main(args)
