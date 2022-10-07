from argparse import ArgumentParser, Namespace
import pandas as pd
import os


def parse_args() -> Namespace:
    """
    python concate_dialog.py \
        -p ./runs/finetune_out/generated_predictions.txt \
        -s ./data/out_of_domain/test/source.csv \
        -o out_t5_out.txt \
        -t ./data/out_of_domain/test/target.csv
    python concate_dialog.py \
        -p ./models/in_domain/grf-in_domain_eval/result_ep:test.txt \
        -s ./data/in_domain/test/source.csv \
        -o out_gpt2_in.txt \
        -t ./data/in_domain/test/target.csv
    python concate_dialog.py \
        -p ./models/out_of_domain/grf-out_of_domain_eval/result_ep:test.txt \
        -s ./data/out_of_domain/test/source.csv \
        -o out_gpt2_out.txt \
        -t ./data/out_of_domain/test/target.csv
    """
    parser = ArgumentParser()
    parser.add_argument(
        "-s",
        "--source",
        type=str,
        default="./data/in_domain/test/source.csv",
    )  # ./data/out_of_domain/test/source.csv
    parser.add_argument(
        "-p",
        "--prediction",
        type=str,
        default="./runs/finetune/generated_predictions.txt",
    )
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
        default="out_t5_in.txt",
    )  # out_{model}_in/out.txt

    args = parser.parse_args()
    return args


def main(args):
    source = pd.read_csv(f"{args.source}", names=["k", "s1", "s2"])
    target = pd.read_csv(f"{args.target}", names=["k", "t1"])
    with open(f"{args.prediction}", 'r') as f:
        pred = f.readlines()
    p = pd.DataFrame(pred, columns=["trans"])
    p['trans'] = p['trans'].apply(lambda x : x.replace("\n", ""))
    p['k'] = 0
    for i in range(0, len(p)):
        if "out_of_domain" in args.target:
            p['k'][i] = i
        else:
            p['k'][i] = i+1
    data=pd.merge(left=source, right=p, how='left', on=['k'])
    data=pd.merge(left=data, right=target, how='left', on=['k'])
    out = data[['s1', 's2', 'trans', 't1']]
    out.to_csv(r'out.txt', header=None, index=None, sep=' ', mode='a')
    # remove " and ' in txt
    with open('out.txt', 'r') as f, open(f"{args.out_txt}", 'w') as fo:
        for line in f:
            fo.write(line.replace('"', '').replace("'", ""))
    os.remove('out.txt')

if __name__ == "__main__":
    args = parse_args()
    main(args)
