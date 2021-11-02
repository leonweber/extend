import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

from spacy import displacy


from pyvis.network import Network
import networkx as nx
import json
import argparse
from pathlib import Path
from flair.tokenization import SegtokSentenceSplitter

from extend.data.processing import restrict_graph_to_span, adapt_span
from extend.data.utils import get_anchors


def plot_example(text: str, G: nx.DiGraph):
    docs = []
    sentences = SegtokSentenceSplitter().split(text)
    for sentence in sentences:
        G_sentence = restrict_graph_to_span(G, sentence.start_pos, sentence.end_pos)
        node_to_token_idx = {}
        words = [{"text": i.text, "tag": "", "lemma": ""} for i in sentence.tokens]
        token_starts = [i.start_pos for i in sentence.tokens]
        for anchor in get_anchors(G_sentence):
            start, end = adapt_span(*G_sentence.nodes[anchor]["span"], token_starts=token_starts)
            words[start]["tag"] = "B-Anchor"
            node_to_token_idx[anchor] = start
            for word in words[start+1:end]:
                word["tag"] = "I-Anchor"

        words.append({"text": "GRAPH: ", "tag": "GRAPH: ", "lemma": "GRAPH :"})
        for node, data in G_sentence.nodes.data():
            if data["label"] != "anchor":
                node_to_token_idx[node] = len(words)
                text = data.get("text", "")
                words.append({"text": node, "tag": data["label"], "lemma": text})

        arcs = []
        for u, v, data in G_sentence.edges.data():
            arcs.append({"start": node_to_token_idx[u], "end": node_to_token_idx[v], "label": data["label"], "dir": "right"})
        docs.append({"words": words, "arcs": arcs})

    displacy.serve(docs, manual=True, options={"compact": False, "add_lemma": True})



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=Path)
    parser.add_argument("i", type=int)
    parser.add_argument("--source", type=int, default=None)
    args = parser.parse_args()

    net = Network(height="1000px", width="3000px")
    with open(args.data) as f:
        lines = f.readlines()
    data = json.loads(lines[args.i - 1])
    G = nx.node_link_graph(data["G"])
    plot_example(text=data["text"], G=G)
