import argparse
import json
from collections import defaultdict
from copy import copy
from pathlib import Path
import random
from typing import Dict, List, Tuple, Type

import pybiopax
from pybiopax import biopax as bp
import networkx as nx
from indra.sources.biopax.processor import BiopaxProcessor
from sklearn.model_selection import train_test_split
from indra.sources.biopax.api import process_model
from indra.statements import validate
from tqdm import tqdm


def get_nodes_by_bp_type(G: nx.DiGraph, BPType: Type[bp.BioPaxObject]):
    return [n for n, d in G.nodes.data("type") if d and d != "str" and issubclass(getattr(bp, d), BPType)]



def get_biopax_graph(model):
    edges = {"pathway_component", "right", "left", "component", "cellular_location",
             "organism", "interaction_type", "feature", "not_feature", "control_type",
             "controlled", "controller", "participant", "product", "template"}
    type_blacklist = {bp.SmallMolecule, bp.FragmentFeature}

    G = nx.MultiDiGraph()
    for obj in model.objects.values():
        if type(obj) in type_blacklist:
            continue
        for attr in dir(obj):
            if attr in edges:
                if obj.uid not in G.nodes:
                    G.add_node(obj.uid, type=type(obj).__name__)
                dsts = getattr(obj, attr)
                if not isinstance(dsts, (list, tuple, set)):
                    dsts = [dsts]
                for dst in dsts:
                    if dst and type(dst) not in type_blacklist:
                        if hasattr(dst, "uid"):
                            dst_name = dst.uid
                        else:
                            dst_name = str(dst)
                        G.add_node(dst_name, type=type(dst).__name__)
                        G.add_edge(obj.uid, dst_name, label=attr)

    for entity in model.get_objects_by_type(bp.PhysicalEntity):
        for db, cuids in BiopaxProcessor._get_processed_xrefs(entity).items():
            if db != "UP":
                continue
            for cuid in cuids:
                G.add_edge(entity.uid, cuid, label=db)


    return G

def get_sif_graph(sif_lines):
    G_sif = nx.MultiDiGraph()
    n_unmappable = 0

    name_to_uniprot = {}
    for line in sif_lines[1:]:
        fields = line.strip().split("\t")
        if len(fields) < 2:
            continue
        if "ProteinReference" in fields[1]:
            name = fields[0]
            for entry in fields[3].split(";"):
                db, cuid = entry.split(":")
                if db == "uniprot":
                    name_to_uniprot[name] = cuid

            uniprot = fields[3].split(":")[1]
            name_to_uniprot[name] = uniprot

    for line in sif_lines[1:]:
        fields = line.strip().split("\t")
        if len(fields) < 2 or "ProteinReference" in fields[1]:
            continue

        if "CHEBI" in fields[0] or "CHEBI" in fields[2]:
            continue
        try:
            u = name_to_uniprot[fields[0]]
            v = name_to_uniprot[fields[2]]
            label = fields[1]
            mediator_ids = fields[6].split(";")
            truncated_mediator_ids = []
            for mediator_id in mediator_ids:
                if not "uniprot" in mediator_id and not "reactome" in mediator_id:
                    mediator_id = mediator_id.split("/")[-1]
                truncated_mediator_ids.append(mediator_id)

            G_sif.add_edge(u, v, label=label, mediator_ids=truncated_mediator_ids)
        except KeyError:
            n_unmappable += 1

    # TODO replace with logging
    print("n_unmappable: ", n_unmappable)
    return G_sif



def get_examples(G: nx.MultiDiGraph, pathway: str):
    examples = []

    G_pw = G.subgraph(nx.descendants(G, source=pathway)).copy()
    reactions = get_nodes_by_bp_type(G_pw, bp.Interaction)

    if len(reactions) < 3:
        return []
    for n_reactions in range(3, len(reactions)-1):
        G_example = G_pw.copy()
        for _, d in G_example.nodes.data():
            d["known"] = False
        for _, _, d in G_example.edges.data():
            d["known"] = False
        known_reactions = random.sample(reactions, n_reactions)
        for reaction in known_reactions:
            for u, v, key in G_example.subgraph(nx.descendants(G_example, source=reaction)).edges:
                G_example.nodes[u]["known"] = True
                G_example.nodes[v]["known"] = True
                G_example.edges[u, v, key]["known"] = True

        examples.append(G_example)

    return examples

def enrich_graph_with_sif(G: nx.MultiDiGraph,
                          mediator_to_edges: Dict[str, List[Tuple]]
                          ) -> None:
    for node in list(G.nodes):
        if node in mediator_to_edges:
            edges = mediator_to_edges[node]
            for u, v, key, data in edges:
                if u not in G.nodes or v not in G.nodes:
                    continue
                if (u, v, key) in G.edges and G.nodes[node]["known"]:
                    G.edges[u, v, key]["known"] = True
                else:
                    data = copy(data)
                    G.add_edge(u, v, key=key, **data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--biopax", default="", required=False, type=Path)
    parser.add_argument("--sif", default="", required=False, type=Path)
    parser.add_argument("--output", default="", required=False, type=Path)

    args = parser.parse_args()
    args.biopax = "/glusterfs/dfs-gfs-dist/weberple/PathwayCommons12.reactome.BIOPAX.owl"
    args.sif = "/glusterfs/dfs-gfs-dist/weberple/PathwayCommons12.reactome.hgnc.txt"
    args.output = Path("/glusterfs/dfs-gfs-dist/weberple")

    with open(args.biopax) as f:
        owl_str = f.read().replace("rdf:about", "rdf:ID")

    model = pybiopax.model_from_owl_str(owl_str)
    G = get_biopax_graph(model)
    pathways = get_nodes_by_bp_type(G, bp.Pathway)

    with open(args.sif) as f:
        sif_lines = f.readlines()

    G_sif = get_sif_graph(sif_lines)
    mediator_to_edges = defaultdict(list)
    for u, v, key in G_sif.edges:
        for mediator in G_sif.edges[u, v, key]["mediator_ids"]:
            data = G_sif.edges[u, v, key]
            mediator_to_edges[mediator].append((u, v, key, data))

    train, rest = train_test_split(pathways, train_size=0.6)
    dev, test = train_test_split(rest, train_size=(1/0.4 * 0.1))

    args.output.mkdir(exist_ok=True)

    with (args.output / "train.jsonl").open("w") as f:
        for pathway in tqdm(train): # we leave it to the training algorithm to decide how to blank out nodes and edges
            G_pw = G.subgraph(nx.descendants(G, source=pathway)).copy()
            for n, d in G_pw.nodes.data():
                d["known"] = True
            for _, _, d in G_pw.edges.data():
                d["known"] = True

            enrich_graph_with_sif(G=G_pw, G_sif=G_sif)
            line = {"G": nx.node_link_data(G_pw)}
            f.write(json.dumps(line))

    with (args.output / "dev.jsonl").open("w") as f:
        for pathway in tqdm(dev[:10]):
            Gs_pw = get_examples(G, pathway)
            for G_pw in Gs_pw:
                enrich_graph_with_sif(G=G_pw, mediator_to_edges=mediator_to_edges)
                line = {"G": nx.node_link_data(G_pw)}
                f.write(json.dumps(line) + "\n")

    with (args.output / "test.jsonl").open("w") as f:
        for pathway in tqdm(test):
            Gs_pw = get_examples(G, pathway)
            for G_pw in Gs_pw:
                enrich_graph_with_sif(G=G_pw, G_sif=G_sif)
                line = {"G": nx.node_link_data(G_pw)}
                f.write(json.dumps(line))
