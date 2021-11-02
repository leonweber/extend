from networkx.algorithms.isomorphism import DiGraphMatcher
from extend.eval.predict_extend import edge_match, node_match
from pyvis.network import Network
import networkx as nx
import json
import argparse
from pathlib import Path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=Path)
    parser.add_argument("i", type=int)
    args = parser.parse_args()

    net = Network(height="1000px", width="3000px")
    with open(args.data) as f:
        lines = f.readlines()
    data = json.loads(lines[args.i - 1])
    G_true = nx.node_link_graph(data["G"])
    G_pred = nx.node_link_graph(data["G_pred"])
    matcher = DiGraphMatcher(G_true, G_pred, node_match=node_match,
                             edge_match=edge_match)

    largest_subgraph_isomorphism = {}
    for isomorphism in matcher.subgraph_isomorphisms_iter():
        if len(isomorphism) > len(largest_subgraph_isomorphism):
            largest_subgraph_isomorphism = isomorphism

    G = nx.DiGraph()

    for node, node_data in G_true.nodes(data=True):
        if node_data["known"]:
            node_data["color"] = "black"
        else:
            if node in largest_subgraph_isomorphism:
                node_data["color"] = "green"
            else:
                node_data["color"] = "red"
        node_data["value"] = 10
        if "text" in node_data:
            node_data["label"] = node_data["text"]
        else:
            node_data["label"] = node_data["type"]

        if node_data["type"] == "anchor":
            node_data["label"] = f"anchor(\"{node_data['label']}\")"

        G.add_node(node, **node_data)

    for u, v, edge_data in G_true.edges(data=True):
        edge_data["label"] = edge_data["type"]
        G.add_edge(u, v, **edge_data)

    for node, node_data in G_pred.nodes(data=True):
        node_data["color"] = None
        node_data["value"] = 10
        if "text" in node_data:
            node_data["label"] = node_data["text"]
        else:
            node_data["label"] = node_data["type"]

        if node_data["type"] == "anchor":
            node_data["label"] = f"anchor(\"{node_data['label']}\")"

        G.add_node("pred_" + node, **node_data)

    for u, v, edge_data in G_pred.edges(data=True):
        edge_data["label"] = edge_data["type"]
        u = "pred_" + u
        v = "pred_" + v
        G.add_edge(u, v, **edge_data)

    G.add_node(data["text"], value=0.1, color=None, label=data["text"])
    net.from_nx(G)
    net.show("foo.html")
