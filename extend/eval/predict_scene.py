import pickle

import hydra
import numpy as np
import torch.cuda
from tqdm import tqdm
import pytorch_lightning as pl

from extend.data.dataset import SceneGraphModificationDataset, BaseDataset
from extend.models.baselines import CopySource


def load_data(output_file):
    with open(output_file, "rb") as fr:
        while True:
            try:
                yield pickle.load(fr)
            except EOFError:
                    break


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


def get_nodes_and_edges_from_orig(G):
    nodes = set()
    edges = set()

    node_to_name = {}
    for node, data in G.nodes(data=True):
        node_to_name[node] = data["feature"]
        nodes.add(data["feature"])

    for u, v, data in G.edges(data=True):
        u = node_to_name[u]
        v = node_to_name[v]

        edge = sorted([u, v])
        edges.add((edge[0], data["feature"], edge[1]))

    return nodes, edges


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

    src_path = str(hydra.utils.to_absolute_path(config["input"])) + "_src_graph.bin"
    tgt_path = str(hydra.utils.to_absolute_path(config["input"])) + "_tgt_graph.bin"
    txt_path = str(hydra.utils.to_absolute_path(config["input"])) + "_src_text.txt"
    with open(txt_path) as f:
        texts = f.readlines()

    Gs_src = list(load_data(src_path))
    Gs_tgt = list(load_data(tgt_path))

    correct = []
    correct_conserved = []

    pbar = tqdm(list(zip(Gs_src, Gs_tgt, texts)))
    f1_nodes = F1()
    f1_edges = F1()
    for i, (G_src, G_tgt, text) in enumerate(pbar):
        nodes_src, edges_src = get_nodes_and_edges_from_orig(G_src)
        G_src = SceneGraphModificationDataset.prepare_src_graph(G_src)
        try:
            G_pred = model.predict(text=text,
                                   G_partial=G_src,
                                   linearize_graph=SceneGraphModificationDataset.linearize_graph,
                                   beam_size=1)
            nodes_pred, edges_pred = SceneGraphModificationDataset.get_nodes_and_edges(G_pred)
        except (ValueError, IndexError):
            nodes_pred = set()
            edges_pred = set()
        nodes_true, edges_true = get_nodes_and_edges_from_orig(G_tgt)

        if (nodes_true == nodes_pred) and (edges_true == edges_pred):
            correct.append(1)
        else:
            correct.append(0)

        conserved_nodes = nodes_src & nodes_true
        conserved_edges = edges_src & edges_true

        if conserved_edges.issubset(edges_pred) and conserved_nodes.issubset(nodes_pred):
            correct_conserved.append(1)
        else:
            correct_conserved.append(0)
        
        f1_nodes.update(nodes_pred, nodes_true)
        f1_edges.update(edges_pred, edges_true)
        
        pbar.set_description(f"acc: {np.mean(correct)*100:.2f}, F1-N: {f1_nodes.get()[2]*100:.2f}, F1-E: {f1_edges.get()[2]*100:.2f}, conserved_acc: {np.mean(correct_conserved)*100:.2f}")
        
if __name__ == '__main__':
    predict()