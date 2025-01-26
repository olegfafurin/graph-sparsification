import random
import math
import logging

import networkx as nx
import igraph as ig
import numpy as np
import os
import re

from community_experiments_plots import reproduce_results, get_similarities
from datasets import initialPath, get_abcd_dataset, get_high_school_dataset, get_primary_school_dataset, \
    get_Amazon_dataset, get_DBLP_dataset
from graph_builder import get_graphs
from clustering import get_partitions
from metrics import get_partitions_similarity_ARI
from community_experiments_plots import build_plots_approx
from graph_builder import get_graphs_approx
from metric_backbone import get_approximate_metric_backbone_igraph


def random_root(G, _, used_roots):
    new_root = random.choice(list(range(G.number_of_nodes())))
    while new_root in used_roots:
        new_root = random.choice(list(range(G.number_of_nodes())))
    return new_root


def min_degree_root(G, csp, used_roots):
    min_degree = min(G.degree(), key=lambda x: x[1])[1]
    candidates = list(G[G.degree == min_degree])
    random.shuffle(candidates)
    for candidate in candidates:
        if candidate not in used_roots:
            return candidate
    return random_root(G, csp, used_roots)


def furthest_from_last_root(G, csp, used_roots):
    if csp is None:
        return random_root(G, csp, used_roots)
    for i, csp_i in sorted(enumerate(csp), key=lambda p: len(p[1]) if len(p[1]) > 0 else G.number_of_nodes(),
                           reverse=True):
        if i not in used_roots:
            return i
    return random_root(G, csp, used_roots)


def compare_approx(get_dataset, out, n_approx_f=None, strategies=tuple("random"), n_iter=5):
    G, D, partition_Meta, B = get_graphs_approx(get_dataset, True, True)
    n = D.number_of_nodes()
    n_clusters = len(set(partition_Meta.values()))

    strategy_d = {"random": random_root, "min_deg": min_degree_root, "furthest": furthest_from_last_root}

    for strategy in strategies:
        f = open(f'{out[:out.rfind(".")]}-{strategy}-{n_iter}-iter.txt', "w")

        strategy_f = strategy_d[strategy] if strategy in strategy_d else random_root
        if n_approx_f is None:
            n_approx = min(3 * n_clusters, n)
        else:
            n_approx = min(n_approx_f(n), n)

        partitions_approx = list()
        for i_iter in range(n_iter):
            partitions_D = get_partitions(D, n_clusters, algos=['Louvain', 'Leiden'])
            get_similarities(f, "Original", partitions_D, partitions_D, partition_Meta, i_iter=i_iter,
                             edge_count=D.number_of_edges())
            partitions_G = get_partitions(G, n_clusters, weight='contact', algos=['Louvain', 'Leiden'])
            get_similarities(f, "Unweighted", partitions_G, partitions_D, partition_Meta, i_iter=i_iter,
                             edge_count=G.number_of_edges(), weighted=False)
            partitions_B = get_partitions(B, n_clusters, algos=['Louvain', 'Leiden'])
            get_similarities(f, "Backbone", partitions_B, partitions_D, partition_Meta, i_iter=i_iter,
                             edge_count=B.number_of_edges())

        D_ig = ig.Graph.from_networkx(D)
        G_ig = ig.Graph.from_networkx(G)
        for base_g, ig_g, weight_path, weight_partition, attrs in [[G, G_ig, 'contact', 'contact', ['contact']],
                                                                   [D, D_ig, 'weight', 'proximity',
                                                                    ['proximity', 'weight']]]:
            G_approx = nx.Graph(base_g)
            d_ig_to_nx = ig_g.vs['_nx_name']
            d_nx_to_ig = {d_ig_to_nx[i]: i for i in range(G_ig.vcount())}
            for i_iter in range(n_iter):
                G_approx.clear_edges()
                used_roots = set()
                csp = None
                for i in range(1, n_approx + 1):
                    root_nx = strategy_f(G_approx, csp, used_roots)
                    used_roots.add(root_nx)
                    root_ig = d_nx_to_ig[root_nx]
                    csp = ig_g.get_shortest_paths(root_ig, weights=weight_path)
                    for p in csp:
                        G_approx.add_edges_from(
                            [(d_ig_to_nx[p[i - 1]], d_ig_to_nx[p[i]],
                              {attr: base_g[d_ig_to_nx[p[i - 1]]][d_ig_to_nx[p[i]]][attr] for attr in attrs}) for i in
                             range(1, len(p))])
                    partition = get_partitions(G_approx, n_clusters, weight=weight_partition,
                                               algos=['Louvain', 'Leiden'])
                    partitions_approx.append(partition)
                    get_similarities(f, f'Approx_{i}_SPTs', partition, partitions_D, partition_Meta, i_iter=i_iter,
                                     edge_count=G_approx.number_of_edges(), weighted=weight_path != 'contact')
                    f.flush()
        f.close()
    return


def compare_abcd(dirname, strategies=tuple("random"), n_iter=5):
    G_contact = nx.read_edgelist(initialPath + f'Datasets/{dirname}/edge_only.txt', nodetype=int,
                                 data=(('contact', int),))
    label_filename = initialPath + f'Datasets/{dirname}/com.dat'
    label = list(map(lambda line: int(line.split()[-1]) - 1, open(label_filename).readlines()))
    idxToLabel = {i: label[i] for i in range(len(label))}
    get_abcd = lambda: (G_contact, idxToLabel)
    out = initialPath + f'Datasets/{dirname}/approx_MB_ari.txt'
    compare_approx(get_abcd, out, strategies=strategies, n_iter=n_iter)


def output_real():
    get_datasets = [get_high_school_dataset, get_primary_school_dataset, get_DBLP_dataset, get_Amazon_dataset]
    outs = [initialPath + path for path in
            ["tmp/high_school_approx.txt", "tmp/primary_school_approx.txt", "tmp/DBLP_approx.txt",
             "tmp/amazon_approx.txt"]]

    for (get_dataset, out) in zip(get_datasets, outs):
        G, D, partition_Meta, B = get_graphs_approx(get_dataset, True, True)
        with open(f'{out[:out.rfind(".")]}-d-edges.txt', "w") as out_contact:
            out_contact.write(f"{D.number_of_nodes()} {D.number_of_edges()}\n")
            for (u, v) in D.edges():
                out_contact.write(f"{u} {v} {D[u][v]['weight']}\n")
        with open(f'{out[:out.rfind(".")]}-com.txt', "w") as out_labels:
            for i in range(D.number_of_nodes()):
                out_labels.write(f"{i} {partition_Meta[i]}\n")


def compare_real():
    get_datasets = [get_high_school_dataset, get_primary_school_dataset, get_DBLP_dataset, get_Amazon_dataset]
    outs = [initialPath + path for path in
            ["tmp/high_school_approx1.txt", "tmp/primary_school_approx1.txt", "tmp/DBLP_approx.txt",
             "tmp/amazon_approx.txt"]]

    get_datasets = get_datasets[:2]
    outs = outs[:2]

    for (get_dataset, out) in zip(get_datasets, outs):
        # compare_approx(get_dataset, out, strategies=["random"])
        compare_approx(get_dataset, out, strategies=["random", "furthest", "min_deg"])


def compare_abcd_ari_vs_k(dirname):
    # ls = os.listdir(dirname)
    # ns = list(filter(lambda x: x is not None, [re.fullmatch('^edge_only_(\\d+)\\.txt$', s) for s in ls]))
    # ns = sorted(list(map(lambda x: int(x[1]), ns)))
    # for k in ns:
    for k in [2, 5, 10, 20]:
        G_contact = nx.read_edgelist(f'{dirname}/edge_only_{k}.txt', nodetype=int, data=(('contact', int),))
        label_filename = f'{dirname}/com_{k}.dat'
        label = list(map(lambda line: int(line.split()[-1]) - 1, open(label_filename).readlines()))
        idxToLabel = {i: label[i] for i in range(len(label))}
        get_abcd = lambda: (G_contact, idxToLabel)
        out = f'{dirname}/ari_k={k}.txt'
        compare_approx(get_abcd, out, n_approx_f=lambda _: 100, strategies=['random'])


def compare_abcd_ari_vs_xi(dirname):
    for xi in list(map(lambda x: str(round(x, 1)), np.arange(0.1, 1.1, 0.1))):
        G_contact = nx.read_edgelist(f'{dirname}/edge_only_{xi}.txt', nodetype=int, data=(('contact', int),))
        label_filename = f'{dirname}/com{xi}.txt'
        label = list(map(lambda line: int(line.split()[-1]) - 1, open(label_filename).readlines()))
        idxToLabel = {i: label[i] for i in range(len(label))}
        get_abcd = lambda: (G_contact, idxToLabel)
        out = f'{dirname}/ari_xi={xi}.txt'
        compare_approx(get_abcd, out, n_approx_f=lambda _: 100, strategies=['random'])


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    # compare_abcd_ari_vs_k("../Datasets/abcd-n-spt/vs_k/n=10000,xi=0.2")
    # compare_abcd_ari_vs_xi("../Datasets/abcd-n-spt/vs_xi/non-regular_n=10000_k=10")

    # compare_abcd("abcd-larger-deg/n=10000")
    # compare_abcd("abcd-larger-deg/n=20000")
    # compare_abcd("abcd-larger-deg/n=30000")

    # output_real()
    compare_real()

    # compare_abcd("abcd/n=1000", strategies=["furthest", "random", "min_deg"])
    # compare_abcd("abcd/n=2000", strategies=["furthest", "random", "min_deg"])
    # compare_abcd("abcd/n=5000", strategies=["furthest", "random", "min_deg"])
    # compare_abcd("abcd/n=10000", strategies=["furthest", "random", "min_deg"])
    # compare_abcd("abcd/n=20000", strategies=["furthest", "random", "min_deg"])

    # dirs = next(os.walk(initialPath + 'Datasets/abcd'))[1]
    # ns = sorted([int(re.fullmatch('^n=(\d+)$', s)[1]) for s in dirs])
    # for n in ns:
    #     dir = f'n={n}'
    #     logging.info(f'processing {dir}...')
    #     compare_abcd(f'abcd/{dir}')

    # reproduce_results()
    # compare_approx("abcd/n=1000")
    # compare_abcd("abcd-larger-deg/n=5000", strategies=["furthest", "random", "min_deg"])
    # compare_abcd("abcd-larger-deg/n=10000", strategies=["furthest", "random", "min_deg"])
    # compare_abcd("abcd-larger-deg/n=20000", strategies=["furthest", "random", "min_deg"])
    # compare_abcd("abcd-larger-deg/n=30000", strategies=["furthest", "random", "min_deg"])
    # build_plots_approx(True)


def test():
    D, partition_Meta, B, T, S = get_graphs(lambda: get_abcd_dataset("n=1000"), True, True)
    B1 = nx.read_edgelist("../Datasets/abcd/n=1000/edge_proximity_mb_full.txt", nodetype=int,
                          data=(('proximity', float),))

    for u, v in B.edges:
        if 'weight' in B[u][v]:
            del B[u][v]['weight']
        B[u][v]['proximity'] = round(B[u][v]['proximity'], 1)

    # nx.write_edgelist(B, "../tmp/B.txt", data=['proximity'])
    # nx.write_edgelist(B1, "../tmp/B1.txt", data=['proximity'])

    B_approx = get_approximate_metric_backbone_igraph(D)

    partitionsB = get_partitions(B, 9)
    partitionsB1 = get_partitions(B1, 9)
    partitionsB_approx = get_partitions(B_approx, 9)
    print(get_partitions_similarity_ARI(partition_Meta, partitionsB['Leiden'][0]))
    print(get_partitions_similarity_ARI(partition_Meta, partitionsB1['Leiden'][0]))
    print(get_partitions_similarity_ARI(partition_Meta, partitionsB_approx['Leiden'][0]))
