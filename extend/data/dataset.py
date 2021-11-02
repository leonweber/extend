import json
import os
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple, Callable, Optional, Dict
from abc import ABC, abstractmethod
from uuid import uuid4

import gdown
import networkx as nx
import pickle

from indra import statements
from tqdm import tqdm

import extend
from extend.data.parse_standoff import StandoffAnnotation, EventTrigger
from extend.data import utils
from extend.data.utils import unpack_file, get_unused_node_name, get_children_by_label, parse_bool, apply_modification_nodes, cached_path


def bionlp_ann_to_graph(ann: StandoffAnnotation):
    G = ann.event_graph.copy()
    for node, data in list(G.nodes(data=True)):
        data["known"] = True
        if "type" in data:
            data["label"] = data["type"]
            del data["type"]

    for _, _, data in G.edges(data=True):
        data["known"] = True
        if "type" in data:
            data["label"] = data["type"]
            del data["type"]

    # We handle modifications such as negation and speculation as special nodes
    for node, data in list(G.nodes(data=True)):
        if "modifications" in data:
            for mod in data["modifications"]:
                mod_name = get_unused_node_name(mod, G)
                G.add_node(mod_name, label=f"MOD_{mod}", known=True)
                G.add_edge(mod_name, node, label="Theme", known=True)
            del data["modifications"]

    # Entities are known and are treated exactly as events
    # They receive a entity node and an anchor node
    for entity_trigger in ann.entity_triggers:
        entity_trigger = entity_trigger.id
        anchor_name = get_unused_node_name("anchor", G)
        data = G.nodes[entity_trigger]
        G.add_node(
            anchor_name, label="anchor", text=data["text"], span=data["span"], known=True
        )
        del data["span"]
        G.add_edge(entity_trigger, anchor_name, label="self", known=True)
        G.nodes[entity_trigger]["known"] = True

    # Rename event triggers to anchors because they fulfill exactly this role in our abstraction
    for event_trigger in [
        i for i, v in ann.triggers.items() if isinstance(v, EventTrigger)
    ]:
        G.nodes[event_trigger]["label"] = "anchor"
        anchor_name = get_unused_node_name("anchor", G)
        nx.relabel_nodes(G, {event_trigger: anchor_name}, copy=False)
    for u, v, data in G.edges(data=True):
        if data["label"] == "Trigger":
            data["label"] = "self"

    return G


class BaseDataset(ABC):
    @classmethod
    @abstractmethod
    def download(cls, name: Optional[str] = None):
        """
        Download the dataset and transform it to ggen format
        """
        pass

    @classmethod
    @abstractmethod
    def linearize_graph(
        cls,
        G: nx.DiGraph,
        node_order: Callable[[nx.DiGraph], List[str]],
        edge_order: Callable[[nx.DiGraph], List[Tuple[str, str]]],
    ) -> str:
        """
        Produce a string representation of G. Required for models that operate on such representations.
        """
        pass



class SceneGraphModificationDataset(BaseDataset):
    DATA_URL = "https://drive.google.com/file/d/1K2lo1Dt7GJskyUVR9x5LH-mZya28KcDY/view?usp=sharing"
    base_path = extend.cache_root / "datasets" / "scene"

    def __init__(self, name):
        if name not in {"crowdsourced", "GCC", "mscoco"}:
            raise ValueError("name must be one of 'crowdsourced', 'GCC', 'mscoco'")
        self.name = name


        self.train_data = self.base_path / f"{name}_train.jsonl"
        self.dev_data = self.base_path / f"{name}_dev.jsonl"
        self.test_data = self.base_path / f"{name}_test.jsonl"

        if not (self.train_data.exists() and self.dev_data.exists() and self.test_data.exists()):
            self.download(name)

    @classmethod
    def download(cls, name=None):

        if not cls.base_path.exists():
            os.makedirs(cls.base_path, exist_ok=True)

        gdown.download(
            "https://drive.google.com/u/0/uc?id=1K2lo1Dt7GJskyUVR9x5LH-mZya28KcDY&export=download",
            str(cls.base_path / "data.tgz"),
        )
        unpack_file(cls.base_path / "data.tgz", cls.base_path)

        cls.transform(
            cls.base_path / "data" / "crowdsourced_data" / "train",
            cls.base_path / "crowdsourced_train.jsonl",
        )
        cls.transform(
            cls.base_path / "data" / "crowdsourced_data" / "dev",
            cls.base_path / "crowdsourced_dev.jsonl",
        )
        cls.transform(
            cls.base_path / "data" / "crowdsourced_data" / "test",
            cls.base_path / "crowdsourced_test.jsonl",
        )

        cls.transform(
            cls.base_path / "data" / "GCC_data" / "train", cls.base_path / "GCC_train.jsonl"
        )
        cls.transform(
            cls.base_path / "data" / "GCC_data" / "dev", cls.base_path / "GCC_dev.jsonl"
        )
        cls.transform(
            cls.base_path / "data" / "GCC_data" / "test", cls.base_path / "GCC_test.jsonl"
        )

        cls.transform(
            cls.base_path / "data" / "mscoco_data" / "train",
            cls.base_path / "mscoco_train.jsonl",
        )
        cls.transform(
            cls.base_path / "data" / "mscoco_data" / "dev", cls.base_path / "mscoco_dev.jsonl"
        )
        cls.transform(
            cls.base_path / "data" / "mscoco_data" / "test", cls.base_path / "mscoco_test.jsonl"
        )

    @classmethod
    def transform(cls, path_in: Path, path_out: Path):
        data = cls.read(str(path_in))
        path_out = Path(path_out)
        with path_out.open("w") as f:
            for G, text in data:
                datapoint = {"G": nx.node_link_data(G), "text": text}
                f.write(json.dumps(datapoint) + "\n")

    @classmethod
    def prepare_src_graph(cls, G_src: nx.DiGraph) -> nx.DiGraph:
        """
        Add anchor nodes to G_src
        Returns copy
        Factored out because it is also needed at prediction time
        """
        G = nx.DiGraph()
        for node, d in G_src.nodes(data=True):
            G.add_node(
                G_src.nodes[node]["feature"],
                label="anchor",
                text=d["feature"],
                known=True,
            )

        for n, d in G_src.nodes(data=True):
            G.add_node(
                f"node({d['feature']})", label="node", text=d["feature"], known=True
            )
            G.add_edge(f"node({d['feature']})", d["feature"], label="self", known=True)
            for u, v, edge_data in G_src.edges(n, data=True):
                name_u = f"node({G_src.nodes[u]['feature']})"
                name_v = f"node({G_src.nodes[v]['feature']})"
                if not (name_v, name_u) in G.edges:
                    G.add_edge(name_u, name_v, label=edge_data["feature"], known=True)

        return G

    @classmethod
    def read(cls, path: str) -> List[Tuple[nx.DiGraph, str]]:
        graphs_and_texts = []

        n_total = 0
        n_impossible = 0
        n_edge_mods = 0

        def load_data(output_file):
            with open(output_file, "rb") as fr:
                while True:
                    try:
                        yield pickle.load(fr)
                    except EOFError:
                        break

        src_path = path + "_src_graph.bin"
        tgt_path = path + "_tgt_graph.bin"
        txt_path = path + "_src_text.txt"
        with open(txt_path) as f:
            texts = f.readlines()

        Gs_src = list(load_data(src_path))
        Gs_tgt = list(load_data(tgt_path))

        for G_src, G_tgt, text in tqdm(list(zip(Gs_src, Gs_tgt, texts))):
            try:
                src_nodes = set(d["feature"] for _, d in G_src.nodes(data=True))
                tgt_nodes = set(d["feature"] for _, d in G_tgt.nodes(data=True))

                added_nodes = tgt_nodes - src_nodes
                deleted_nodes = src_nodes - tgt_nodes

                src_edges = set(
                    (G_src.nodes[u]["feature"], G_src.nodes[v]["feature"])
                    for u, v in G_src.edges()
                )
                tgt_edges = set(
                    (G_tgt.nodes[u]["feature"], G_tgt.nodes[v]["feature"])
                    for u, v in G_tgt.edges()
                )

                added_edges = tgt_edges - src_edges
                deleted_edges = src_edges - tgt_edges

                for u, v in added_edges:
                    if u not in added_nodes and v not in added_nodes:
                        n_edge_mods += 1

                for u, v in deleted_edges:
                    if u not in deleted_nodes and v not in deleted_nodes:
                        n_edge_mods += 1

                G = cls.prepare_src_graph(G_src)

                for n, d in G_tgt.nodes(data=True):
                    n = d["feature"]
                    if n in G.nodes:
                        continue

                    G.add_node(n, label="anchor", known=False, text=n)
                    assert n in text
                    start = text.index(n)
                    end = start + len(n)
                    G.nodes[n]["span"] = (start, end)

                for deleted_node in deleted_nodes:
                    node = f"del({deleted_node})"
                    G.add_node(node, label="DELETE", text=node, known=False)
                    G.add_edge(node, f"node({deleted_node})", label="theme", known=False)

                for added_node in added_nodes:
                    n_total += 1
                    added_node_ids = [
                        i
                        for i, d in G_tgt.nodes(data=True)
                        if d["feature"] == added_node
                    ]
                    if added_node.lower() not in text.lower():
                        n_impossible += 1
                    assert len(added_node_ids) == 1
                    node = f"add({added_node})"
                    G.add_node(node, label="ADD", text=node, known=False)
                    for u, v, edge_data in G_tgt.edges(added_node_ids[0], data=True):
                        v = G_tgt.nodes[v]["feature"]
                        if v in added_nodes:
                            v = f"add({v})"
                        else:
                            v = f"node({v})"
                        if (v, node) not in G.edges:
                            G.add_edge(node, v, label=edge_data["feature"], known=False)
                    G.add_edge(node, added_node, label="theme", known=False)
            except AssertionError:
                n_impossible += 1
                continue

            try:
                nx.find_cycle(G)
                n_impossible += 1
                continue
            except nx.NetworkXNoCycle:
                graphs_and_texts.append((G, text))

        return graphs_and_texts

    @classmethod
    def get_node_text(cls, node: str, G: nx.DiGraph) -> str:
        data = G.nodes[node]
        if "text" in data:
            return data["text"]
        elif data["label"] in {"ADD", "DELETE"}:
            if data["label"] == "ADD":
                op_name = "add"
            elif data["label"] == "DELETE":
                op_name = "del"
            else:
                raise ValueError

            args = [
                n for _, n, d in G.out_edges(node, data=True) if d["label"] == "theme"
            ]
            args_str = ", ".join(cls.get_node_text(n, G) for n in args)

            return f"{op_name}({args_str})"

        else:
            return data["label"]

    @classmethod
    def linearize_graph(cls, G, node_order, edge_order):
        linearization = ""

        for node in node_order(G):
            if G.nodes[node]["label"] == "anchor":
                continue
            if G.nodes[node]["label"] in {"ADD", "DELETE"}:
                continue
            if f"node({node})" in G.nodes:
                continue

            linearization += "| "
            start = len(linearization)
            linearization += cls.get_node_text(node, G)
            end = len(linearization)
            G.nodes[node]["span2"] = (start, end)

        for u, v in edge_order(G):
            if G.get_edge_data(u, v)["label"] == "self":
                continue
            if G.nodes[u]["label"] in {"ADD", "DELETE"} or G.nodes[v]["label"] in {
                "ADD",
                "DELETE",
            }:
                continue
            rel = G.edges[u, v]["label"]
            u = cls.get_node_text(u, G)
            v = cls.get_node_text(v, G)
            linearization += "| " + f"{u} {rel} {v}"

        for node in node_order(G):
            if G.nodes[node]["label"] not in {"ADD", "DELETE"}:
                continue

            linearization += "| "
            start = len(linearization)
            linearization += cls.get_node_text(node, G)
            end = len(linearization)
            G.nodes[node]["span2"] = (start, end)

        return linearization

    @classmethod
    def linearize_graph_no_edges(cls, G, node_order, edge_order):
        linearization = ""

        for node in node_order(G):
            if G.nodes[node]["label"] == "anchor":
                continue
            if G.nodes[node]["label"] in {"ADD", "DELETE"}:
                continue
            if f"node({node})" in G.nodes:
                continue

            linearization += "| "
            start = len(linearization)
            linearization += cls.get_node_text(node, G)
            end = len(linearization)
            G.nodes[node]["span2"] = (start, end)

        for node in node_order(G):
            if G.nodes[node]["label"] not in {"ADD", "DELETE"}:
                continue

            linearization += "| "
            start = len(linearization)
            linearization += cls.get_node_text(node, G)
            end = len(linearization)
            G.nodes[node]["span2"] = (start, end)

        return linearization

    @classmethod
    def strip_node(self, node):
        if "add(" in node:
            node = node.replace("add(", "").replace(")", "")
        if "node(" in node:
            node = node.replace("node(", "").replace(")", "")

        return node

    @classmethod
    def get_nodes_and_edges(cls, G: nx.DiGraph):
        G_resulting = apply_modification_nodes(G)

        nodes = set()
        for node in G_resulting.nodes:
            if "node" in node:
                nodes.add(cls.strip_node(node))

        edges = set()
        for u, v, data in G_resulting.edges(data=True):
            u = cls.strip_node(u)
            v = cls.strip_node(v)

            if data["label"] in {"self", "theme"}:
                continue

            edge = sorted([u, v])
            edges.add((edge[0], data["label"], edge[1]))

        return nodes, edges



class BioNLPCompletionDataset(BaseDataset):
    name_to_links = {
        "pc13": "https://drive.google.com/uc?id=1lHbll4xl6nFZBPACNHBnTWLDG2dAAwE2"
    }
    base_path = extend.cache_root / "datasets" / "bionlp_completion"

    def __init__(self, name):
        self.name = name

        self.train_data = self.base_path / f"{name}_train.jsonl"
        self.dev_data = self.base_path / f"{name}_dev.jsonl"
        self.test_data = self.base_path / f"{name}_test.jsonl"

        if not (self.train_data.exists() and self.dev_data.exists() and self.test_data.exists()):
            self.download(name)

    @classmethod
    def download(cls, name=None):
        if name not in cls.name_to_links:
            raise ValueError("name should be one of " + ", ".join(cls.name_to_links))

        if not cls.base_path.exists():
            os.makedirs(cls.base_path, exist_ok=True)

        gdown.download(
            cls.name_to_links[name],
            str(cls.base_path / f"{name}.zip"),
        )
        unpack_file(cls.base_path / f"{name}.zip", cls.base_path)


    @classmethod
    def get_node_text(cls, node, G, edge_order):
        if "text" in G.nodes[node]:
            node_text = G.nodes[node]["text"]
        else:
            node_text = G.nodes[node]["label"]
        text = node_text
        for u, v in edge_order(G):
            if u == node and G.nodes[v]["label"] != "anchor":
                text += f" {G.edges[(u, v)]['label']}({cls.get_node_text(v, G, edge_order)})"

        return text

    @classmethod
    def get_node_text_natural_language(cls, node, G, edge_order):
        if "text" in G.nodes[node]:
            node_text = G.nodes[node]["text"]
        else:
            node_text = G.nodes[node]["label"]
        text = node_text
        causes = [cls.get_node_text_natural_language(v, G, edge_order) for u, v, d in G.out_edges(node, data=True) if d["label"] == "Cause"]
        themes = [cls.get_node_text_natural_language(v, G, edge_order) for u, v, d in G.out_edges(node, data=True) if d["label"] == "Theme"]
        sites = [cls.get_node_text_natural_language(v, G, edge_order) for u, v, d in G.out_edges(node, data=True) if d["label"] == "Site"]
        from_locs = [cls.get_node_text_natural_language(v, G, edge_order) for u, v, d in G.out_edges(node, data=True) if d["label"] == "FromLoc"]
        to_locs = [cls.get_node_text_natural_language(v, G, edge_order) for u, v, d in G.out_edges(node, data=True) if d["label"] == "ToLoc"]
        at_locs = [cls.get_node_text_natural_language(v, G, edge_order) for u, v, d in G.out_edges(node, data=True) if d["label"] == "AtLoc"]
        participants = [cls.get_node_text_natural_language(v, G, edge_order) for u, v, d in G.out_edges(node, data=True) if d["label"] == "Participant"]

        if themes:
            text += " of " + " and ".join(themes)
        if causes:
            text += " by " + " and ".join(causes)
        if sites:
            text += " at " + " and ".join(sites)
        if from_locs:
            text += " from " + " and ".join(from_locs)
        if to_locs:
            text += " to " + " and ".join(to_locs)
        if at_locs:
            text += " at " + " and ".join(at_locs)
        if participants:
            text += " with " + " and ".join(participants)

        return text

    @classmethod
    def linearize_graph_natural_language(
            cls,
            G: nx.DiGraph,
            node_order: Callable[[nx.DiGraph], List[str]],
            edge_order: Callable[[nx.DiGraph], List[Tuple[str, str]]],
                                         ) -> str:
        linearization = ""
        for node in node_order(G):
            if G.nodes[node]["label"] == "anchor":
                continue
            start = len(linearization)
            linearization += "| " + cls.get_node_text_natural_language(node, G, edge_order=edge_order)
            end = len(linearization)
            G.nodes[node]["span2"] = (start, end)

        return linearization


    @classmethod
    def linearize_graph(
            cls,
            G: nx.DiGraph,
            node_order: Callable[[nx.DiGraph], List[str]],
            edge_order: Callable[[nx.DiGraph], List[Tuple[str, str]]],
    ) -> str:
        linearization = ""
        for node in node_order(G):
            if G.nodes[node]["label"] == "anchor":
                continue
            start = len(linearization)
            linearization += "| " + cls.get_node_text(node, G, edge_order=edge_order)
            end = len(linearization)
            G.nodes[node]["span2"] = (start, end)

        return linearization


