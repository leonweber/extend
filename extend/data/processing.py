import json
import warnings
from bisect import bisect_right, bisect_left
from collections import defaultdict, Counter
from operator import itemgetter
from typing import List, Dict, Optional, Callable
import logging

import networkx as nx
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from extend.data.transforms import Transform, IdentityTransform
from extend.data.tokenization import SegtokSentenceSplitter, Sentence
from extend.data.utils import Dictionary, overlaps, get_dependants

log = logging.getLogger(__name__)

def adapt_span(start, end, token_starts):
    """
    Adapt annotations to token spans
    """
    start = int(start)
    end = int(end)

    new_start = bisect_right(token_starts, start) - 1
    new_end = bisect_left(token_starts, end)

    return new_start, new_end



class Text2GraphProcessing:
    def __init__(self, path: str, transform: Optional[Transform] = None,
                 small: bool = False):
        self.transform = transform or IdentityTransform()
        self.examples = []
        self.small = small

        with open(path) as f:
            for lino, line in enumerate(f.readlines()):
                if lino > 100 and self.small:
                    break
                example_orig = json.loads(line)
                if not "text" in example_orig:
                    example_orig["text"] = ""
                G_orig = nx.node_link_graph(example_orig["G"])
                for G_transformed in self.transform(G_orig):
                    example_transformed = {"G": G_transformed, "text": example_orig["text"]}
                    for k, v in example_orig.items():
                        if k not in {"G", "text"}:
                            example_transformed[k] = v
                    self.examples.append(example_transformed)

        self._node_label_count = Counter()
        self._edge_label_count = Counter()

        for example in self.examples:
            self._node_label_count.update(d["label"] for _, d in example["G"].nodes(data=True))
            self._edge_label_count.update(d["label"] for _, _, d in example["G"].edges(data=True))
        self.node_labels = sorted(self._node_label_count.keys())
        self.edge_types = sorted(self._edge_label_count.keys())
        self.print_statistics()

    def __getitem__(self, item):
        return self.examples[item]

    def __len__(self):
        return len(self.examples)

    def print_statistics(self):
        logging.info(f"Num. examples: {len(self)}")
        logging.info(f"Num. node labels: {len(self.node_labels)}")
        logging.info(f"Num. edge labels: {len(self.edge_types)}")



def restrict_graph_to_span(G_full: nx.DiGraph, start: int, end: int) -> nx.DiGraph:
    """
    Only retain the nodes whose anchors overlap with span and adapt the anchors' spans
    to be correct relative to the span.

    @param G_full:
    @param start:
    @param end:
    @return:
    """

    G_full = G_full.copy()
    anchors_to_delete = set()
    for node, data in G_full.nodes(data=True):
        if data["label"] == "anchor":
            if "span" in data and not overlaps((start, end), data["span"]):
                anchors_to_delete.add(node)
            else:
                if "span" in data:
                    data["span"] = (int(data["span"][0]) - start,
                                    int(data["span"][1]) - start)

    nodes_to_delete = anchors_to_delete.copy()
    for node in anchors_to_delete:
        nodes_to_delete.update(get_dependants(node, G_full))


    G_restricted = G_full.subgraph([i for i in G_full.nodes if i not in nodes_to_delete])

    return G_restricted


class LinearBertProcessing(Text2GraphProcessing):
    def __init__(self, path: str,
                 linearize_graph: Callable[[nx.DiGraph], str],
                 tokenizer: str,
                 max_len: int = 128,
                 node_label_dict: Optional[Dictionary] = None,
                 edge_label_dict: Optional[Dictionary] = None,
                 transform: Optional[Transform] = None,
                 include_eog: bool = True,
                 split_sentences: bool = False,
                 remove_rare_node_types: int = 0,
                 remove_rare_edge_types: int = 5,
                 small: bool = False):
        super(LinearBertProcessing, self).__init__(path, transform=transform, small=small)
        self.max_len = max_len
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, use_fast=True)
        self.max_events_per_sentence = 10
        self.linearize_graph = linearize_graph
        self.include_eog = include_eog
        self.remove_rare_node_types = remove_rare_node_types
        self.remove_rare_edge_types = remove_rare_edge_types
        if split_sentences:
            self.sentence_splitter = SegtokSentenceSplitter()
        else:
            self.sentence_splitter = None

        if edge_label_dict is None:
            self.edge_label_dict = Dictionary()
            self.edge_label_dict.add_item("O")
            self.edge_label_dict.add_item("<unk>")
            for edge_type in self.edge_types:
                if self._edge_label_count[edge_type] >= self.remove_rare_edge_types:
                    self.edge_label_dict.add_item("B-" + edge_type)
                    self.edge_label_dict.add_item("I-" + edge_type)
        else:
            self.edge_label_dict = edge_label_dict

        if node_label_dict is None:
            self.node_label_dict = Dictionary()
            self.node_label_dict.add_item("None")
            self.node_label_dict.add_item("<unk>")
            for node_type in self.node_labels:
                if self._node_label_count[node_type] >= self.remove_rare_node_types:
                    self.node_label_dict.add_item(node_type)
        else:
            self.node_label_dict = node_label_dict

        self._n_skipped_examples = 0
        new_examples = []
        for datapoint in tqdm(self.examples, desc="Building examples"):
            new_examples += self.transform_datapoint(datapoint)
        warnings.warn(f"Skipped {self._n_skipped_examples} examples because of errors (most likely because the input was truncated).")
        self.examples = new_examples

        self.print_statistics_unrolled()

    def print_statistics_unrolled(self):
        logging.info(f"Num. unrolled examples: {len(self)}")
        logging.info(f"Num. unrolled node types: {len(self.node_label_dict)}")
        logging.info(f"Num. unrolled edge types: {len(self.edge_label_dict)}")

    def transform_datapoint(self, datapoint: Dict) -> List[Dict[str, torch.Tensor]]:
        examples = []
        G_full = datapoint["G"]
        text = datapoint["text"]
        if self.sentence_splitter:
            sentences = self.sentence_splitter.split(text)
        else:
            sentences = [Sentence(text=text, start_position=0)]
        for i, sentence in enumerate(sentences):
            if not sentence.text.strip():
                continue
            start = sentence.start_position
            try:
                end = sentences[i+1].start_position
            except IndexError:
                end = len(text) + 1
            G_sentence = restrict_graph_to_span(G_full, start=start, end=end)

            known_nodes = [i for i, d in G_sentence.nodes(data=True) if d["known"]]
            # warnings.filterwarnings("error")

            for node in self.node_order(G_sentence):
                if node not in known_nodes and G_sentence.nodes[node]["label"] != "anchor":
                    try:
                        examples.append(self.build_example(node=node, G_full=G_sentence,
                                                           known_nodes=known_nodes, text1=sentence.text,
                                                           tokenizer=self.tokenizer,
                                                           edge_label_dict=self.edge_label_dict,
                                                           node_label_dict=self.node_label_dict,
                                                           linearize_graph=self.linearize_graph,
                                                           allow_truncation=False,
                                                           max_len=self.max_len
                                                           ))
                    except Exception as e:
                        if isinstance(e, (InterruptedError, KeyboardInterrupt)):
                            raise e
                        else:
                            self._n_skipped_examples += 1
                    known_nodes.append(node)
                    for _, v, d in G_sentence.out_edges(node, data=True):
                        if d["label"] == "self" and G_sentence.nodes[v]["label"] == "anchor":
                            known_nodes.append(v) # add anchor nodes

            if self.include_eog:
                try:
                    examples.append(self.build_example(node=None, G_full=G_sentence,
                                                       known_nodes=known_nodes, text1=sentence.text,
                                                       tokenizer=self.tokenizer,
                                                       edge_label_dict=self.edge_label_dict,
                                                       node_label_dict=self.node_label_dict,
                                                       linearize_graph=self.linearize_graph,
                                                       allow_truncation=False,
                                                       max_len=self.max_len
                                                       ))
                except Exception as e:
                    if isinstance(e, (InterruptedError, KeyboardInterrupt)):
                        raise e
                    else:
                        self._n_skipped_examples += 1

            warnings.filterwarnings("default")

        return examples

    @staticmethod
    def node_order(G):
        G = G.copy()
        order = []
        zero_outdegree = [u for u, d in G.out_degree() if d == 0]

        while zero_outdegree:
            sort_tuple_to_events = defaultdict(list)
            for node in zero_outdegree:
                sort_tuple = []
                sort_tuple.append(G.nodes[node]["label"])
                sort_tuple_to_events[tuple(sort_tuple)].append(node)
                G.remove_node(node)

            for sort_tuple in sorted(sort_tuple_to_events, reverse=True):
                order += sort_tuple_to_events[sort_tuple]

            zero_outdegree = [u for u, d in G.out_degree() if d == 0]

        return order

    @classmethod
    def build_example(cls, node, G_full, known_nodes, text1, tokenizer,
                      node_label_dict, edge_label_dict, linearize_graph,
                      allow_truncation, max_len):
        G_known = G_full.subgraph(known_nodes)
        text2 = linearize_graph(G_known, node_order=cls.node_order,
                                edge_order=cls.edge_order)
        encoding = tokenizer.encode_plus(text1, text2, padding="max_length",
                                              truncation="only_first",
                                              return_tensors="pt",
                                              return_offsets_mapping=True,
                                              max_length=max_len,
                                              return_length=True)

        if allow_truncation:
            warnings.simplefilter("always", UserWarning)
        else:
            warnings.simplefilter("error", UserWarning)
        for input_ids in encoding["input_ids"]:
            if not 0 in input_ids:
                warnings.warn("Text was truncated")
                break
        warnings.simplefilter("once", UserWarning)

        for k, v in encoding.items():
            encoding[k] = v.squeeze()
        seq2_start = torch.where(encoding["offset_mapping"].sum(dim=1) == 0)[0][1].item()
        pad_start = torch.where(encoding["offset_mapping"].sum(dim=1) == 0)[0][2].item()
        token_starts1 = encoding["offset_mapping"][:seq2_start, 0].tolist()
        token_starts2 = encoding["offset_mapping"][seq2_start:pad_start, 0].tolist()
        labels = torch.zeros_like(encoding["input_ids"])

        if node is not None:
            labels[0] = node_label_dict.get_idx_for_item(G_full.nodes[node]["label"])  # Predict node type with [CLS]
            for _, v, d in G_full.out_edges(node, data=True):
                if "span2" in G_full.nodes[v]:
                    span = G_full.nodes[v]["span2"]
                    token_starts = token_starts2
                    offset = seq2_start
                else:
                    span = G_full.nodes[v]["span"]
                    token_starts = token_starts1
                    offset = 0

                span = adapt_span(span[0], span[1], token_starts)
                labels[span[0] + offset] = edge_label_dict.get_idx_for_item("B-" + d["label"])
                for i in range(span[0] + 1 + offset, span[1] + offset):
                    labels[i] = edge_label_dict.get_idx_for_item("I-" + d["label"])

        encoding["labels"] = labels

        return encoding


    @classmethod
    def edge_order(cls, G):
        nodes = cls.node_order(G)
        sort_tuples = []
        for u, v in G.edges():
            idx = max(nodes.index(u), nodes.index(v))
            sort_tuples.append((idx, (u, v)))
        sort_tuples = sorted(sort_tuples, key=itemgetter(0))

        return [i[1] for i in sort_tuples]




