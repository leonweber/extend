import argparse
import json

import numpy as np
from extend.data import transforms
from extend.data.dataset import get_processor
from extend.data.processing import Text2GraphProcessing
from extend.predict.utils import apply_modification_nodes


def print_statistics(examples):
    n_nodes_per_example_orig = []
    n_edges_per_example_orig = []
    n_nodes_per_example_mod = []
    n_edges_per_example_mod = []
    n_tokens = 0

    for example in examples:
        G = example["G"].copy()
        G_orig = G.subgraph(i for i, d in G.nodes.data() if d["type"] not in ["ADD", "DEL", "anchor"] and d["known"])
        G_mod = apply_modification_nodes(G)
        G_mod = G_mod.subgraph(i for i, d in G_mod.nodes.data() if d["type"] != "anchor")
        n_nodes_per_example_orig.append(len(G_orig.nodes))
        n_edges_per_example_orig.append(len(G_orig.edges))
        n_nodes_per_example_mod.append(len(G_mod.nodes))
        n_edges_per_example_mod.append(len(G_mod.edges))
        n_tokens += len(example["text"].split())

    n_examples = len(examples)
    print(f"{n_examples} examples")
    print()
    # print("Distribution of nodes:")
    # print(Counter(n_nodes_per_example).most_common())
    # print()
    # print("Distribution of edges:")
    # print(Counter(n_edges_per_example).most_common())
    print("Num nodes orig.")
    print(f"{np.mean(n_nodes_per_example_orig):.2f} +/- {np.std(n_nodes_per_example_orig):.2f}")
    print()
    print("Num edges orig.")
    print(f"{np.mean(n_edges_per_example_orig):.2f} +/- {np.std(n_edges_per_example_orig):.2f}")
    print()
    print("Num nodes mod.")
    print(f"{np.mean(n_nodes_per_example_mod):.2f} +/- {np.std(n_nodes_per_example_mod):.2f}")
    print()
    print("Num edges mod.")
    print(f"{np.mean(n_edges_per_example_mod):.2f} +/- {np.std(n_edges_per_example_mod):.2f}")
    print()
    print(f"{n_tokens/len(examples):.2f} tokens per example")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/bionlp_speculation.json")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    processor = get_processor(config["data"])

    if config["transform"]:
        transform = getattr(transforms, config["transform"])()
    else:
        transform = None

    train = Text2GraphProcessing(processor.train_data, transform=transform)
    dev = Text2GraphProcessing(processor.dev_data, transform=transform)
    test = Text2GraphProcessing(processor.test_data, transform=transform)

    train_tokens = set()
    for example in train.examples:
        train_tokens.update(example["text"].split())
    dev_tokens = set()
    for example in dev.examples:
        dev_tokens.update(example["text"].split())
    test_tokens = set()
    for example in test.examples:
        test_tokens.update(example["text"].split())

    train_nodes = set()
    for example in train.examples:
        train_nodes.update(i for _, i in example["G"].nodes().data("text"))
    dev_nodes = set()
    for example in dev.examples:
        dev_nodes.update(i for _, i in example["G"].nodes().data("text"))
    test_nodes = set()
    for example in test.examples:
        test_nodes.update(i for _, i in example["G"].nodes().data("text"))



    print("===TRAIN===")
    print("------------")
    print_statistics(train.examples)
    print()
    print()
    print("===DEV===")
    print("------------")
    print_statistics(dev.examples)
    print()
    print(f"OOV DEV Text: {100*len(dev_tokens - train_tokens) / len(dev_tokens):.2f}%")
    print(f"OOV DEV Nodes: {100*len(dev_nodes - train_nodes) / len(dev_nodes):.2f}%")
    print()
    print()
    print("===TEST===")
    print("------------")
    print_statistics(test.examples)
    print()
    print()
    print("===TOTAL===")
    print("------------")
    print_statistics(train.examples + dev.examples + test.examples)
    print()
    print(f"OOV TEST Text: {100*len(test_tokens - train_tokens) / len(test_tokens):.2f}%")
    print(f"OOV TEST Nodes: {100*len(test_nodes - train_nodes) / len(test_nodes):.2f}%")




