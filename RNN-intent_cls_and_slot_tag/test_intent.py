import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
from tqdm import tqdm

from utils import Vocab
from model_intent import SeqClassifier
from dataset_intent import SeqClsDataset


def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())

    torch.manual_seed(1)  # seed
    test_data = None
    with open(args.test_file, "r") as fp:
        test_data = json.load(fp)
    test_dataset = SeqClsDataset(test_data, vocab, intent2idx, args.max_len)
    # TODO: create DataLoader for test dataset
    test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False, batch_size=150, collate_fn=test_dataset.collate_fn)

    embeddings = torch.load(args.cache_dir / "embeddings.pt")

    # load weights into model
    model = SeqClassifier(
        embeddings=embeddings,
        hidden_size=args.hidden_size,
        dropout=args.dropout,
        num_layers=args.num_layers,
        bidirectional=args.bidirectional,
        num_class=test_dataset.num_classes)
    device = args.device

    try:
        ckpt = torch.load(args.ckpt_path)  # "./ckpt/intent/model.ckpt"
        model.load_state_dict(ckpt)
    except:
        pass
    model.to(device)
    model.eval()

    # predict
    preds = []
    ids = [d['id'] for d in test_data]
    with open(args.pred_file, "w") as fp:
        fp.write("id,intent\n")
        with torch.no_grad():
            for i, d in enumerate(tqdm(test_loader)):
                out, _ = model(d["text"].to("cpu"), None)  # cuda:0
                _, pred = torch.max(out, 1)
                for j, p in enumerate(pred):
                    fp.write(f"{ids[150*i+j]},{test_dataset.idx2label(p.item())}\n")


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--test_file",
        type=Path,
        help="Path to the test file.",
        required=True,
        default="./data/intent/test.json"  # 新增預設
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/intent/",
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        help="Path to model checkpoint.",
        required=True,
        default="./ckpt/intent/model.ckpt"  # 新增預設
    )
    parser.add_argument("--pred_file", type=Path, default="pred.intent.csv")

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)  # 256
    parser.add_argument("--num_layers", type=int, default=2)  # 2
    parser.add_argument("--dropout", type=float, default=0.4)  # 0.2
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
    main(args)
