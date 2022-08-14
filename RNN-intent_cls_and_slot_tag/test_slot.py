import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
from tqdm import tqdm

from utils import Vocab
from model_slot import SeqClassifier
from dataset_slot import SeqClsDataset


def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    tag_idx_path = args.cache_dir / "tag2idx.json"
    tag2idx: Dict[str, int] = json.loads(tag_idx_path.read_text())
    torch.manual_seed(1)
    test_data = None
    with open(args.test_file, "r") as fp:
        test_data = json.load(fp)
    test_dataset = SeqClsDataset(test_data, vocab, tag2idx, args.max_len)
    # TODO: create DataLoader for test dataset
    test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False, batch_size=150, collate_fn=test_dataset.collate_fn)

    embeddings = torch.load(args.cache_dir / "embeddings.pt")
    model = SeqClassifier(
        embeddings=embeddings,
        hidden_size=args.hidden_size,
        dropout=args.dropout,
        num_layers=args.num_layers,
        bidirectional=args.bidirectional,
        num_class=10)
    device = args.device

    try:
        ckpt = torch.load(args.ckpt_path)  # "./ckpt/slot/model.ckpt"
        model.load_state_dict(ckpt)
    except:
        pass

    model.to(device)
    model.eval()
    ids = [d["id"] for d in test_data]

    # TODO: predict dataset
    preds = []
    ids = [d['id'] for d in test_data]
    with open(args.pred_file, "w") as fp:
        fp.write("id,tags\n")
        with torch.no_grad():
            for i, d in enumerate(tqdm(test_loader)):
                out, _ = model(d["tokens"].to("cpu"), None)  # cuda:1
                _, pred = torch.max(out, 2)
                for j, p in enumerate(pred):
                    fp.write(f"{ids[150*i+j]},{' '.join(list(map(lambda x:test_dataset.idx2label(x), list(filter(lambda x: (x != 9), p.tolist())))))}\n")


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--test_file",
        type=Path,
        help="Path to the test file.",
        default="./data/slot/test.json"
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/slot/",
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        help="Path to model checkpoint.",
        required=True
    )
    parser.add_argument(
        "-p",
        "--pred_file",
        type=Path,
        default="./pred.slot.csv"
    )

    # data
    parser.add_argument("--max_len", type=int, default=64)

    # model
    parser.add_argument("--hidden_size", type=int, default=1024)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.5)  # 0.4 next 0.5
    parser.add_argument("--bidirectional", type=bool, default=True)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    print(args)
    main(args)
