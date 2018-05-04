#!/usr/bin/python3
"""Analysis of network motifs."""
import numpy as np
import typing
import networkx as nx
import itertools


def motif_counter(gr, mo, k):
    """Count motifs in a directed graph.

    Args:
        gr: `DiGraph` object
        dicmo: `dict` of motifs to count
        k: An `int` specifying motif length
    Returns:
        mcount: `dict` with the number of each motifs, with the same keys as ``mo``

    This function is actually rather simple. It will extract all 3-grams from
    the original graph, and look for isomorphisms in the motifs contained
    in a dictionary. The returned object is a ``dict`` with the number of
    times each motif was found.::
        >>> print mcounter(gr, mo)
        {'S1': 4, 'S3': 0, 'S2': 1, 'S5': 0, 'S4': 3}
    """
    # From: https://gist.github.com/tpoisot/8582648
    #This function will take each possible subgraph of gr of size 3, then
    #compare them to the mo dict using .subgraph() and is_isomorphic
    #This line simply creates a dictionary with 0 for all values, and the
    #motif names as keys
    mcount = dict(zip(mo.keys(), list(map(int, np.zeros(len(mo))))))
    nodes = gr.nodes()

    #We use iterools.product to have all combinations of three nodes in the
    #original graph. Then we filter combinations with non-unique nodes, because
    #the motifs do not account for self-consumption.
    triplets = list(itertools.product(*[nodes]*k))
    triplets = [trip for trip in triplets if len(list(set(trip))) == k]
    triplets = map(list, map(np.sort, triplets))
    u_triplets = []
    [u_triplets.append(trip) for trip in triplets if not u_triplets.count(trip)]

    #The for each each of the triplets, we (i) take its subgraph, and compare
    #it to all the possible motifs
    for trip in u_triplets:
        sub_gr = gr.subgraph(trip)
        mot_match = list(map(lambda mot_id: nx.is_isomorphic(sub_gr, mo[mot_id]), mo.keys()))
        match_keys = [list(mo.keys())[i] for i in range(len(mo)) if mot_match[i]]
        if len(match_keys) == 1:
            mcount[match_keys[0]] += 1
    return mcount


def generate_motifs(k: int) -> typing.Dict[str, nx.DiGraph]:
    """Generate all topologically unique, directed graphs of size `k`."""
    combs = list(itertools.permutations(range(1,k+1), 2)) # all node combinations = all possible edges
    motifs = {}
    cnt = -1
    for kk in range(1,k+1):  # number of edges in motif
        combscombs = list(itertools.combinations(combs, kk))  # all node combinations with kk edges
        for cc in combscombs:  # now filter motifs
            # make sure the motif contains all nodes
            a = []
            [a.extend(list(c)) for c in cc]  # flatten list
            all_edges_contained = np.unique(a).shape[0]>=k
            # make sure each edge exists only once in one direction in the motif
            aa = [c.sort() for c in list(cc)]  # list of all sorted (undirected) edges
            all_edges_unique = len(aa)==np.unique(aa, axis=0).shape[0]  # True only if all edges are unique
            # make sure the motif is a fully connected graph
            fully_connected = len(aa)>=k-1
            # check if a topologically similar motif is alread in list
            mot_match = list(map(lambda mot_id: nx.is_isomorphic(nx.DiGraph(list(cc)), motifs[mot_id]), motifs.keys()))
            motif_is_new = not(any(mot_match))

            # only add motif if it matches all of the above criteria
            if all_edges_contained and all_edges_unique and motif_is_new and fully_connected:
                cnt += 1
                motifs[f"S{k}-{cnt}"] = nx.DiGraph(list(cc))
    return motifs


def generate_chain(chain_length):
    """Make a linear (chain) graph of a given size."""
    G = nx.DiGraph()
    for edg in range(chain_length):
        G.add_edge(edg, edg+1)
    return G


def generate_circle(circle_size):
    """Make a circular graph of a given size.

    A circle is a chain with the end connected to the start.
    """
    G = generate_chain(circle_size)
    G.add_edge(circle_size, 0)
    return G


def generate_special_motifs(kmin: int, kmax: int = 10) -> typing.Dict[str, nx.DiGraph]:
    """Generate all circle and chain graphs with at least `kmin` and at most `kmax` nodes.

    Args:
        kmin: Minimal number of nodes in graphs
        kmax: Maximum number of nodes in grapsh
    Returns:
        dict: Dictionary of graphs
    """
    special_motifs = dict()
    for k in range(kmin, kmax):
        special_motifs[f"chain{k+1}"] = generate_chain(k)
        special_motifs[f"circle{k+1}"] = generate_circle(k)
    return special_motifs


def process_motifs(chainer, chainee, max_k=4):
    """Count all motifs of size `max_k` in the network defined by chainer->chainee edges.

    Args:
        chainer: `list` [frames x ...]
        chainer: `list` [frames x ...]
        max_k=4:
    Returns:
        motif_counts: `list` of counts for each motif in `motifs`
        motifs: `dict`
    """
    # autogenerate network motifs up to size max_k
    motifs = [0]*(max_k+1)
    nframes = chainer.shape[1]
    motif_counts_all = []
    for k in range(2,max_k+1):
        motifs[k] = generate_motifs(k)
        print(f"There exist {len(motifs[k])} topologically unique, directed motifs of size {k}")
    motifs.append(generate_special_motifs(max_k, 10))

    # motifs in all frames
    nmotifs = sum([len(b) for b in motifs[2:]])
    motif_counts = np.zeros((nmotifs, nframes), dtype=np.uint16)
    # initialize - add all flies as nodes
    for frame_number in range(nframes):
        if frame_number % 10000 == 0:
            print(frame_number)

        G = nx.DiGraph()
        # add chaining connections
        this_chainer = chainer[chainer[:, frame_number]>=0, frame_number]
        this_chainee = chainee[chainer[:, frame_number]>=0, frame_number]
        for tcher, tchee in zip(this_chainer, this_chainee):
            G.add_edge(tcher, tchee)

        motif_counts_this_all = np.zeros((0,), dtype=np.uint16)
        for k in range(2, min(len(this_chainer)+1, max_k+1)):
            this_motif_counts = motif_counter(G, motifs[k], k)
            motif_counts_this_all = np.concatenate( (motif_counts_this_all, list(this_motif_counts.values())))
        motif_counts[:len(motif_counts_this_all),frame_number] = motif_counts_this_all
    return motif_counts, motifs
