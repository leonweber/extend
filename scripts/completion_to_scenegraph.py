import argparse
from pathlib import Path
import os
import pickle

import networkx as nx
from extend.data.utils import get_G_known
from extend.data.processing import Text2GraphProcessing

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    os.makedirs(args.out.parent, exist_ok=True)

    dataset = Text2GraphProcessing(args.data)

    with open(str(args.out) + "_src_graph.bin", "wb") as f_src, open(
        str(args.out) + "_tgt_graph.bin", "wb"
    ) as f_tgt, open(str(args.out) + "_src_text.txt", "w") as f_txt:
        for example in dataset:
            G_src = get_G_known(example["G"]).copy()
            for _, d in G_src.nodes(data=True):
                if "text" in d:
                    d["feature"] = d["text"]
                else:
                    d["feature"] = d["type"]

            for _, _, d in G_src.edges(data=True):
                d["feature"] = d["type"]

            G_tgt = example["G"].copy()
            for _, d in G_tgt.nodes(data=True):
                if "text" in d:
                    d["feature"] = d["text"]
                else:
                    d["feature"] = d["type"]

            for _, _, d in G_tgt.edges(data=True):
                d["feature"] = d["type"]


            G_src.add_node("<start>", feature="<start>") # SceneGraph cannot handle empty graphs
            G_src = nx.convert_node_labels_to_integers(G_src.to_undirected())
            pickle.dump(G_src, f_src)

            G_tgt.add_node("<start>", feature="<start>") # SceneGraph cannot handle empty graphs
            G_tgt = nx.convert_node_labels_to_integers(G_tgt.to_undirected())
            pickle.dump(G_tgt.to_undirected(), f_tgt)
            f_txt.write(example["text"].strip() + "\n")




