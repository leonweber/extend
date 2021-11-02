import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Optional

import networkx as nx
from tqdm import tqdm

from extend.data.dataset import bionlp_ann_to_graph
from extend.data.processing import restrict_graph_to_span
from extend.data.tokenization import SegtokSentenceSplitter
from extend.data.utils import get_unused_node_name, get_dependants, get_anchors
from extend.data.parse_standoff import StandoffAnnotation, EventTrigger, EVENT_TRIGGER_TYPES




def graph_to_completion_graph(
    G: nx.DiGraph,
    n_max_events_delete: int,
) -> Optional[nx.DiGraph]:
    G = G.copy()

    G.remove_nodes_from(get_anchors(G))

    events = [n for n, d in G.nodes(data=True) if d["type"].lower() in EVENT_TRIGGER_TYPES]
    modifications = [n for n, d in G.nodes(data=True) if d["type"].startswith("MOD_")]

    deletable_nodes = events + modifications

    # n_max_events_delete = min(int(len(deletable_nodes) * (2/3)), n_max_events_delete)
    n_max_events_delete = min(len(deletable_nodes), n_max_events_delete)

    if not n_max_events_delete:
        return G

    n_events_delete = random.randint(0, n_max_events_delete)


    nodes_to_delete = set()
    for _ in range(n_events_delete):
        node_to_delete = random.sample(deletable_nodes, 1)[0]
        dependants = get_dependants(node_to_delete, G)
        bunch_to_delete = {node_to_delete} | dependants
        if len(bunch_to_delete) + len(nodes_to_delete) > n_events_delete:
            break
        else:
            nodes_to_delete.update(bunch_to_delete)

    for node in nodes_to_delete:
        G.nodes[node]["known"] = False
        for _, v, edge_data in G.out_edges(node, data=True):
            edge_data["known"] = False
            if edge_data["type"] == "self" and G.nodes[v]["type"] == "anchor":
                G.nodes[v]["known"] = False

    return G


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--n_max_events_delete", default=3, type=int)
    parser.add_argument("--n_examples_per_graph", default=3, type=int)

    args = parser.parse_args()


    sentence_splitter = SegtokSentenceSplitter()

    n_examples = 0
    n_unknown = 0
    n_events = 0
    n_unknown_per_example = []

    with args.out.open("w") as f_out:
        for txt_file in tqdm(list(args.data.glob("*.txt"))):
            with txt_file.open() as f:
                text = f.read().strip()
            with txt_file.with_suffix(".a1").open() as f:
                a1_lines = f.readlines()
            with txt_file.with_suffix(".a2").open() as f:
                a2_lines = f.readlines()

            ann = StandoffAnnotation(a1_lines, a2_lines)
            G = bionlp_ann_to_graph(ann)

            sentences = [s for s in sentence_splitter.split(text) if s.text]
            for i, sentence in enumerate(sentences):
                start = sentence.start_position
                try:
                    end = sentences[i + 1].start_position
                except IndexError:
                    end = len(text) + 1

                G_sentence = restrict_graph_to_span(G, start=start, end=end)
                generated_examples = set()

                for _ in range(args.n_examples_per_graph):
                    G_completion = graph_to_completion_graph(
                        G_sentence,
                        n_max_events_delete=args.n_max_events_delete,
                    )

                    if not G_completion:
                        continue

                    G_known = G_completion.copy().subgraph(n for n, d in G_completion.nodes(data=True) if d["known"])

                    generated_example = json.dumps(
                        {
                            "G": nx.node_link_data(G_completion),
                            "text": sentence.text,
                        })
                    if generated_example not in generated_examples:
                        generated_examples.add(generated_example)
                        f_out.write(generated_example + "\n")
                        n_examples += 1
                        n_events += len([n for n, d in G_completion.nodes(data=True) if d["type"].lower() in EVENT_TRIGGER_TYPES or d["type"].startswith("MOD_")])
                        n_unknown_per_example.append(len([n for n, d in G_completion.nodes(data=True) if (d["type"].lower() in EVENT_TRIGGER_TYPES or d["type"].startswith("MOD_")) and not d["known"]]))
                        n_unknown += n_unknown_per_example[-1]
        print(f"{n_examples} examples with {n_unknown/n_examples} unkown events on average")
        print(f"{100*n_unknown/n_events}% of the {n_events} events are unknown")
        print("Distribution of unknowns over examples:")
        print(Counter(n_unknown_per_example).most_common())
