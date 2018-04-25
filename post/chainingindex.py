"""Detect and count chains."""
import numpy as np
import scipy.spatial
import networkx as nx


def get_chaining(lines, chamber_number=0):
    """Determine interactions: chaining based on head-tail distance and angle, head-butting based on head-head distances and angles."""
    nframes = lines.shape[0]
    nflies = lines.shape[2]
    tails = lines[:, chamber_number, :, 0, :]
    heads = lines[:, chamber_number, :, 1, :]

    D_h2t = np.zeros((nflies, nflies, nframes))
    D_h2h = np.zeros((nflies, nflies, nframes))
    Dc = np.zeros((nflies, nflies, nframes), dtype=np.bool)
    Dh = np.zeros((nflies, nflies, nframes), dtype=np.bool)

    chainee = -np.ones((nflies*nflies, nframes), dtype=np.int16)
    chainer = -np.ones((nflies*nflies, nframes), dtype=np.int16)
    headee = -np.ones((nflies*nflies, nframes), dtype=np.int16)
    header = -np.ones((nflies*nflies, nframes), dtype=np.int16)

    for frame_number in range(0, nframes):
        T = frame_number
        D_h2t[:, :, T] = scipy.spatial.distance.cdist(tails[T, :, :], heads[T, :, :], metric='euclidean')
        D_h2h[:, :, T] = scipy.spatial.distance.cdist(heads[T, :, :], heads[T, :, :], metric='euclidean')

        flylength = np.diag(D_h2t[:, :, T])  # diagonals contain tail->head distances=fly lengths
        min_distance = np.min(flylength)     # interaction distance is flylength+x
        Dc[:, :, T] = D_h2t[:, :, T] < min_distance    # chaining
        Dh[:, :, T] = D_h2h[:, :, T] < min_distance/2  # head-butting
        # ignore diagonal entries
        np.fill_diagonal(Dc[:, :, T], False)
        np.fill_diagonal(Dh[:, :, T], False)

        # get x,y coords of interacting flies
        chainee_this, chainer_this = np.where(Dc[:, :, T])
        headee_this, header_this = np.where(Dh[:, :, T])

        # save all in list
        chainee[0:chainee_this.shape[0], T] = chainee_this
        chainer[0:chainer_this.shape[0], T] = chainer_this
        headee[0:headee_this.shape[0], T] = headee_this
        header[0:header_this.shape[0], T] = header_this
    return chainee, chainer, headee, header, D_h2t, D_h2h, Dc, Dh


def get_chainlength(chainer, chainee, nflies):
    """Identify "chains" (subnetworks) and return size and membership.

    Args:
        chainer: list of chain sources [ x nframes], filled with -1
        chainee: list of chain targets [ x nframes], filled with -1
        nflies:  number of flies
    Returns:
        chain_length: length of each chain [nflies (max number of chains) x nframes]
        chain_id: bool matrix [nflies x nflies (max number of chains) x nframes]. True if fly (axis=0) is member of chain (axis=1)
    """
    nframes = chainer.shape[1]

    chain_length = np.zeros((nflies, nframes), dtype=np.uint16)
    chain_id = np.zeros((nflies, nflies, nframes), dtype=np.bool)
    chain_nedges = np.zeros((nflies, nframes), dtype=np.uint16)
    for frame_number in range(0, nframes):
        edges = [(int(source), int(target)) for source, target in zip(chainer[:,frame_number], chainee[:,frame_number]) if source>0]
        if len(edges):
            G = nx.Graph(edges)
            subgraphs = list(nx.connected_component_subgraphs(G))

            # length of each chain = number of flies in subgraph
            this_chain_len = [subgraph.number_of_nodes() for subgraph in subgraphs]
            chain_length[:len(this_chain_len), frame_number] = this_chain_len

            # number of edges: nodes-1 for chains, >=nodes for circles and jumbles
            this_chain_nedges = [subgraph.number_of_edges() for subgraph in subgraphs]
            chain_nedges[:len(this_chain_nedges), frame_number] = this_chain_nedges

            # which fly is in which chain? make boolean array [nflies, nchains] for each frame
            for this_chain_count, this_chain in enumerate(subgraphs):
                chain_id[list(this_chain), this_chain_count, frame_number] = True

    return chain_length, chain_id, chain_nedges
