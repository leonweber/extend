from typing import Tuple, Set

import hydra
import networkx as nx
import pytorch_lightning as pl
import numpy as np
import torch
from networkx.algorithms.isomorphism import DiGraphMatcher
from tqdm import tqdm

from extend.data.processing import Text2GraphProcessing


def node_match(n1, n2):
    do_match = n1["label"] == n2["label"]

    if "text" in n1:
        do_match = do_match and "text" in n2 and n1["text"] == n2["text"]
    elif "text" in n2:
        do_match = do_match and "text" in n1 and n1["text"] == n2["text"]

    return do_match


def edge_match(e1, e2):
    return e1["label"] == e2["label"]


def get_edges(G: nx.DiGraph) -> Set[Tuple[str, str, str]]:
    edges = set()

    for u, v, data in G.edges.data():
        edges.add((u, data["label"], v))

    return edges

class F1:
    def __init__(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0

    def update(self, pred, true):
        self.tp += len(pred & true)
        self.fp += len(pred - true)
        self.fn += len(true - pred)

    def get(self):
        try:
            p = self.tp / (self.tp + self.fp)
        except ZeroDivisionError:
            p = 0
        try:
            r = self.tp / (self.tp + self.fn)
        except ZeroDivisionError:
            r = 0
        try:
            f1 = 2*p*r / (p+r)
        except ZeroDivisionError:
            f1 = 0

        return p, r, f1

    def __str__(self):
        return f"P: {self.get()[0]*100:.2f}, R: {self.get()[1]*100:.2f}"




@hydra.main(config_path="../../config", config_name="predict")
def predict(config):
    model: pl.LightningModule
    model = hydra.utils.instantiate(config["model"],
                                    edge_label_dict={},
                                    node_label_dict={},
                                    train_size=1)
    model = model.load_from_checkpoint(hydra.utils.to_absolute_path(config["checkpoint"]))
    model.eval()

    if torch.cuda.is_available():
        model.cuda()

    dataset = hydra.utils.instantiate(config["dataset"])

    correct = []
    node_f1 = F1()
    edge_f1 = F1()
    pbar = tqdm(Text2GraphProcessing(dataset.test_data))

    for i, example in enumerate(pbar):
        G_true: nx.DiGraph
        G_true = example["G"]
        G_known = G_true.copy().subgraph(n for n, d in G_true.nodes(data=True) if d["known"])
        try:
            G_pred = model.predict(text=example["text"],
                                   G_partial=G_known,
                                   linearize_graph=dataset.linearize_graph,
                                   beam_size=1).copy()
            anchors = [n for n, d in G_pred.nodes(data=True) if d["label"] == "anchor"]
            G_pred.remove_nodes_from(anchors)
        except Exception as e:
            if isinstance(e, (InterruptedError, KeyboardInterrupt)):
                raise ValueError
            else:
                G_pred = nx.DiGraph()

        anchors = [n for n, d in G_true.nodes(data=True) if d["label"] == "anchor"]
        G_true.remove_nodes_from(anchors)

        correct.append(nx.is_isomorphic(G_pred, G_true, node_match=node_match, edge_match=edge_match))

        pred_extension_nodes = [n for n in G_pred if n not in G_known]
        true_extension_nodes = [n for n in G_true if n not in G_known]
        pred_known_nodes = [n for n in G_pred if n in G_known]

        node_f1.update(G_pred.subgraph(pred_known_nodes).nodes, G_known.nodes)
        edge_f1.update(G_pred.subgraph(pred_known_nodes).edges, G_known.edges)

        for node_pred in pred_extension_nodes:
            descendants = list(nx.single_source_dijkstra(G=G_pred, source=node_pred)[0])
            descendant_subgraph = G_pred.subgraph(descendants)
            matcher = DiGraphMatcher(G_true, descendant_subgraph, node_match=node_match,
                                     edge_match=edge_match)
            if matcher.subgraph_is_isomorphic():
                node_f1.tp += 1
                edge_f1.tp += len(G_pred.out_edges(node_pred)) # If there is a subgraph isomorphism, then all edges are correct
            else:
                node_f1.fp += 1
                edge_f1.fp += len(G_pred.out_edges(node_pred)) # If there is no subgraph isomorphism, then we define all edges to be incorrect

        for node_true in true_extension_nodes:
            descendants = list(nx.single_source_dijkstra(G=G_true, source=node_true)[0])
            descendant_subgraph = G_true.subgraph(descendants)
            matcher = DiGraphMatcher(G_true, descendant_subgraph, node_match=node_match,
                                     edge_match=edge_match)
            if matcher.subgraph_is_isomorphic():
                pass # was already counted when handling predicted extension nodes
            else:
                node_f1.fn += 1
                relabeling = {}
                edge_f1.fn += len(G_true.out_edges(node_true)) # If there is no subgraph isomorphism, then we define all edges to be incorrect

        pbar.set_description(f"acc: {np.mean(correct)*100:.2f}, F1-N: {node_f1.get()[2]*100:.2f}, F1-E: {edge_f1.get()[2]*100:.2f}")
        print(f"Nodes: {node_f1}")
        print(f"Edges: {edge_f1}")

if __name__ == '__main__':
    predict()