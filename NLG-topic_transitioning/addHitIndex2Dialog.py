from argparse import ArgumentParser, Namespace
import json
import spacy
from tqdm import tqdm


def parse_args() -> Namespace:
    """
    python addHitIndex2Dialog.py -d blenderbot_train_4819.jsonl -o out_for_split_train_4819.jsonl
    python addHitIndex2Dialog.py -d blenderbot_val_1000.jsonl -o out_for_split_val.jsonl
    """
    parser = ArgumentParser()
    parser.add_argument(
        "-d",
        "--dialogue",
        type=str,
        default="dialogue.jsonl",
        help="list of dict. the format is in the following"
        """
        [{
            "id": 0,
            "dialog": ["A1", "B1", "A2", "B2",...,"A5", "B5"]
        },{
            "id": 1,
            "dialog": ["C1", "D1", "C2", "D2",...,"C5", "D5"]
        }]
        """
    )
    parser.add_argument(
        "-o",
        "--out_for_split",
        type=str,
        default="out_for_split.jsonl",
        help="list of dict. the format is in the following"
        """
        [{
            "id": 0,
            "dialog": ["A1", "B1", "A2", "B2",...,"A5", "B5"],
            "index": 2,
            "target": "A2"
        },]
        """
    )

    args = parser.parse_args()
    return args


def main(args):
    with open("keywords.json") as f:
        keywords = json.load(f)

    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    # lemmatize words in keywords
    for key, val in keywords.items():
        # separate words by its length (one, others)
        one_lemma = []
        multi_lemma = []
        for word in val:
            split = [token.lemma_ for token in nlp(word)]
            if len(split) >= 2:
                multi_lemma.append(" ".join(split))
            else:
                one_lemma.append(split[0])
            keywords[key] = [one_lemma, multi_lemma]

    with open(args.dialogue, "r") as f:
        dialogue = [json.loads(line) for line in f]

    # add hit keyword index
    for d in tqdm(dialogue):
        for index in range(2, len(d["dialog"]), 2):
            lemma_utterance = [token.lemma_ for token in nlp(d["dialog"][index])]
            for key, (one, multi) in keywords.items():
                intersection = set(one) & set(lemma_utterance)
                for m in multi:
                    unsplit_utterance = " ".join(lemma_utterance)
                    if m in unsplit_utterance:
                        intersection.add(m)
                if len(intersection) > 0:
                    d["index"] = index
                    d["target"] = d["dialog"][index]
                    break
    # add index = -1
    for d in dialogue:
        if "index" not in d.keys():
            d["index"] = -1
            d["target"] = ""

    with open(f"{args.out_for_split}", "w") as f:
        for idx, dialogue in enumerate(dialogue):
            f.write(
                json.dumps(
                    {
                        "id": idx,
                        "dialog": dialogue["dialog"],
                        "index": dialogue["index"],
                        "target": dialogue["target"]
                        }) + "\n")


if __name__ == "__main__":
    args = parse_args()
    main(args)
