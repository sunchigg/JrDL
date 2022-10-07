"""
[T5]
sacrebleu ./data/in_domain/test/target.txt -i ./runs/finetune/generated_predictions.txt -b -m bleu -w 3 --lowercase
sacrebleu ./data/out_of_domain/test/target.txt -i ./runs/finetune_out/generated_predictions.txt -b -m bleu -w 3 --lowercase
[GPT2]
sacrebleu ./data/in_domain/test/target.txt -i ./models/in_domain/grf-in_domain_eval/result_ep:test.txt -b -m bleu -w 3 --lowercase
sacrebleu ./data/out_of_domain/test/target.txt -i ./models/out_of_domain/grf-out_of_domain_eval/result_ep:test.txt -b -m bleu -w 3 --lowercase
"""
from argparse import ArgumentParser, Namespace
import pandas as pd
import os


def parse_args() -> Namespace:
    """
    python csv2txt.py
    python csv2txt.py -t ./data/out_of_domain/test/target.csv -o ./data/out_of_domain/test/target.txt
    """
    parser = ArgumentParser()
    parser.add_argument(
        "-t",
        "--target",
        type=str,
        default="./data/in_domain/test/target.csv",
    )  # ./data/out_of_domain/test/target.csv
    parser.add_argument(
        "-o",
        "--out_txt",
        type=str,
        default="./data/in_domain/test/target.txt",
    )  # out_{model}_in/out.txt

    args = parser.parse_args()
    return args


def main(args):
    target = pd.read_csv(f"{args.target}", names=["k", "t1"])
    target_bleu = target[['t1']]
    target_bleu.to_csv(r'ot.txt', header=None, index=None, sep=' ', mode='a')
    with open('ot.txt', 'r') as f, open(f"{args.out_txt}", 'w') as fo:
        for line in f:
            fo.write(line.replace('"', '').replace("'", ""))
    os.remove('ot.txt')

if __name__ == "__main__":
    args = parse_args()
    main(args)
