import argparse
import json
from pathlib import Path
import os

from extend.data import transforms
from extend.data.processing import LinearBertProcessing
from extend.data.dataset import get_processor
from flair.data import Sentence, Token


def example_to_brat(dataset, example):

    tags = [dataset.node_label_dict.get_item_for_index(example["labels"][0].item())]
    tags += [dataset.edge_label_dict.get_item_for_index(i.item()) for i in example["labels"][1:]]
    input_ids = example["input_ids"].clone()
    if len(input_ids.shape) > 1:
        input_ids = input_ids[0]
    tokens = dataset.tokenizer.convert_ids_to_tokens(input_ids.tolist(),
                                                     skip_special_tokens=False)
    tokens = [i for i in tokens if i != "[PAD]"]
    sentence = Sentence()
    for token, tag in zip(tokens, tags):
        whitespace_before = False
        if "##" in token:
            token = token.replace("##", "")
        else:
            whitespace_before = True
        start = len(sentence.to_original_text()) + 1
        if not whitespace_before:
            start -= 1
        sentence.add_token(Token(token, start_position=start))
        sentence.tokens[-1].add_label("Label", tag)
    a1_lines = []
    for i, span in enumerate(sentence.get_spans("Label")):
        mention = sentence.to_original_text()[span.start_pos:span.end_pos]
        a1_lines.append(
            f"T{i}\t{span.tag} {span.start_pos} {span.end_pos}\t{mention}")

    return sentence.to_original_text(), "\n".join(a1_lines)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    with args.config.open() as f:
        config = json.load(f)

    if config["transform"]:
        transform = getattr(transforms, config["transform"])()
    else:
        transform = None
    processor = get_processor(config["data"])
    linearize_graph = getattr(processor, config["linearize_graph"])
    ds = LinearBertProcessing(
        processor.train_data, config=config,
        linearize_graph=linearize_graph,
        transform=transform,
        include_eog=True,
        small=True
    )
    os.makedirs(args.out, exist_ok=True)
    for i, example in enumerate(ds[:100]):
        txt, ann = example_to_brat(ds, example)
        fname = str(i).zfill(6)
        with open(args.out / f"{fname}.txt", "w") as f:
            f.write(txt)
        with open(args.out / f"{fname}.ann", "w") as f:
            f.write(ann)
