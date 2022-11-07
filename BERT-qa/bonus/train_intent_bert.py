import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from tqdm import trange, tqdm

# from utils import Vocab
from dataset_intent_bert import SeqClsDataset
# from model_intent import SeqClassifier
from transformers import AdamW, BertForSequenceClassification, BertTokenizerFast

TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]


def main(args):
    # with open(args.cache_dir / "vocab.pkl", "rb") as f:
    #     vocab: Vocab = pickle.load(f)

    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())

    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    datasets: Dict[str, SeqClsDataset] = {
        # split: SeqClsDataset(split_data, vocab, intent2idx, args.max_len)
        split: SeqClsDataset(split_data, intent2idx, args.max_len, tokenizer, "train")
        for split, split_data in data.items()
    }
    # TODO: crecate DataLoader for train / dev datasets
    train_loader = torch.utils.data.DataLoader(datasets[TRAIN], batch_size = args.batch_size, shuffle = True, collate_fn = datasets[TRAIN].collate_fn)
    dev_loader = torch.utils.data.DataLoader(datasets[DEV],batch_size = args.batch_size, shuffle = False, collate_fn = datasets[DEV].collate_fn)

    # embeddings = torch.load(args.cache_dir / "embeddings.pt")
    # TODO: init model and move model to target device(cpu / gpu)
    # model = SeqClassifier(embeddings = embeddings, hidden_size = args.hidden_size, dropout = args.dropout, num_layers = args.num_layers, bidirectional = True, num_class = 150)
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=150)
    device = args.device
    """try:
        ckpt = torch.load("./ckpt/intent/model.ckpt")
        model.load_state_dict(ckpt)
    except:
        pass"""
    batch_size = args.batch_size

    # TODO: init optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    best_acc = 0.0

    epoch_pbar = trange(args.num_epoch, desc="Epoch")
    for epoch in epoch_pbar:
        # TODO: Training loop - iterate over train dataloader and update model weights
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        val_loss = 0.0
        val_acc = 0.0
        # h = model.init_hidden(batch_size, device)  # init
        for i, data in enumerate(tqdm(train_loader)):
            data = [torch.tensor(j).to(device) for j in data[1:]] 
            optimizer.zero_grad()
            out = model(input_ids = data[0], token_type_ids = data[1], attention_mask=data[2], labels=data[3])
            loss = out.loss
            _, train_pred = torch.max(out.logits, 1)
            loss.backward()
            optimizer.step()

            train_acc += (train_pred.cpu() == data[3].cpu()).sum().item()
            train_loss += loss.item()
            torch.cuda.empty_cache()
        # TODO: Evaluation loop - calculate accuracy and save model weights
        with torch.no_grad():
            # h = model.init_hidden(batch_size, device)
            model.eval()
            for i, dev_data in enumerate(tqdm(dev_loader)):
                dev_data = [torch.tensor(j).to(device) for j in dev_data[1:]]
                out = model(input_ids = dev_data[0], token_type_ids = dev_data[1], attention_mask=dev_data[2], labels=dev_data[3])

                loss = out.loss
                _, val_pred = torch.max(out.logits, 1)
                val_acc += (val_pred.cpu() == dev_data[3].cpu()).sum().item()
                val_loss += loss.item()

            print(f"Epoch {epoch + 1}: Train Acc: {train_acc / len(train_loader.dataset)}, Train Loss: {train_loss / len(train_loader)}, Val Acc: {val_acc / len(dev_loader.dataset)}, Val Loss: {val_loss / len(dev_loader)}")
            if val_acc >= best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(),"./ckpt/intent/model.ckpt")
                print(f"Save model with acc {val_acc / len(dev_loader.dataset)}")
        # pass # 讓原本空白for loop可以過的pass

    # TODO: Inference on test set
    test_data = None
    with open("./data/intent/test.json", "r") as fp:
        test_data = json.load(fp)
    # test_dataset = SeqClsDataset(test_data, vocab, intent2idx, args.max_len)
    test_dataset = SeqClsDataset(test_data, intent2idx, args.max_len, tokenizer, "test")
    test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False, batch_size=150, collate_fn=test_dataset.collate_fn)
    model.eval()
    # ids = [d["id"] for d in test_data]

    # predict dataset
    preds = []
    ids = [d['id'] for d in test_data]
    with open("./pred.intent.csv", "w") as fp:
        fp.write("id,intent\n")
        with torch.no_grad():
            for i, data in enumerate(tqdm(test_loader)):
                ids=data[0]
                data = [torch.tensor(j).to(device) for j in data[1:]] 
                out = model(input_ids = data[0], token_type_ids = data[1], attention_mask=data[2])
                _, pred = torch.max(out.logits, 1)
                for j, p in enumerate(pred):
                    fp.write(f"{ids[j]},{test_dataset.idx2label(p.item())}\n")


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/intent/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/intent/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/intent/",
    )

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)  # 256
    parser.add_argument("--num_layers", type=int, default=2)  # 2
    parser.add_argument("--dropout", type=float, default=0.4)  # 0.2
    parser.add_argument("--bidirectional", type=bool, default=True)
    # parser.add_argument("--att", action="store_true")
    # parser.add_argument("--att_unit", type=int, default=128)
    # parser.add_argument("--att_hops", type=int, default=16)

    # optimizer
    parser.add_argument("--lr", type=float, default=3e-5)  # 1e-3

    # data loader
    parser.add_argument("--batch_size", type=int, default=64)

    # training
    parser.add_argument(
        "--device", type=torch.device,
        help="cpu, cuda, cuda:0, cuda:1, cuda:2, cuda:3",
        default="cuda:1"
    )
    parser.add_argument("--num_epoch", type=int, default=10)  # next 50

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)
