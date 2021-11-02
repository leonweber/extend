import random
from abc import ABC
from typing import List

import networkx as nx


class Transform(ABC):
    def transform(self, G: nx.DiGraph) -> List[nx.DiGraph]:
        raise NotImplementedError

    def __call__(self, G: nx.DiGraph) -> List[nx.DiGraph]:
        return self.transform(G)



def remove_dependants(node: str, G: nx.DiGraph) -> nx.DiGraph:
    G = G.copy()
    nodes_to_remove = {}

    def helper(node_to_remove: str):
        for _, v in G.in_edges(node):
            nodes_to_remove.add(v)
            helper(v)

    G.remove_nodes_from(nodes_to_remove)

    return G


class IdentityTransform(Transform):
    def transform(self, G: nx.DiGraph) -> List[nx.DiGraph]:
        return [G]


class CompleteOne(Transform):
    def transform(self, G: nx.DiGraph) -> List[nx.DiGraph]:
        transformed_graphs = []

        for node, data in G.nodes(data=True):
            if data["type"] != "anchor" and not data["known"]:
                G_transformed = remove_dependants(node, G) # returns copy
                anchors = [v for _, v, edge_data in G_transformed.out_edges(node, data=True) if edge_data["type"] == "self"]
                for node2, data2 in G_transformed.nodes(data=True):
                    if node2 != node and node2 not in anchors:
                        data2["known"] = True

                transformed_graphs.append(G_transformed)

        return transformed_graphs


class AnchorToGenerative(Transform):
    """
    Remove all anchors and replace the types of the anchored nodes with the anchor text.
    This is used to convert a flavor 0/1 to a flavor 2 task in which there is no anchoring
    at all.
    """

    def transform(self, G: nx.DiGraph) -> List[nx.DiGraph]:
        G_transformed = G.copy()

        for node, node_data in G.nodes(data=True):
            if node_data["type"] == "anchor":
                G_transformed.nodes[node]["type"] = node_data["text"]

        return [G_transformed]


class SubsampleNegative(Transform):
    def __init__(self):
        self.sampling_rate = 0.1

    def transform(self, G: nx.DiGraph) -> List[nx.DiGraph]:
        is_positive = not all(i for _, i in G.nodes.data("known"))

        if is_positive or random.uniform(0, 1) <= self.sampling_rate:
            return [G]
        else:
            return []
