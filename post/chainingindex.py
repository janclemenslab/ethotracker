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
#     # TODO: [-] this will ignore circles since there is no seed - a fly that chains but is not a chainee
#     #       [x] very slow... - will run endlessly if there is a loop
#     nframes = chainer.shape[1]
#
#     chain_length = np.zeros((nflies*2, nframes), dtype=np.uint16)
#     chain_id = [None]*nframes
#     chain_id = np.zeros((nflies, nflies*2, nframes), dtype=np.bool)
#     for frame_number in range(0, nframes):
#         # if frame_number % 10000==0:
#         #     print(frame_number)
#         # find starters of chain - flies that are chainer but not chainees
#         chain_seeds = [x for x in chainer[:, frame_number] if x not in chainee[:, frame_number]]
#         # for each starter - follow the chain to the end
#         chain = [-1]*len(chain_seeds)
#         for chain_count, chain_seed in enumerate(chain_seeds):
#             chain[chain_count] = [chain_seed]
#             chain_link = [chain_seed]
#             idx = 1
#             while len(chain_link):
#                 idx, = np.where(chainer[:, frame_number] == chain_link)  # find chainee for current chainer
#                 chain_link = chainee[idx, frame_number].tolist()  # this is the next chainee
#                 # TODO: remove already counted chainers so we don't count them multiple times
#                 try:
#                     if chain_link[0] not in chain[chain_count]:
#                         chain[chain_count].append(chain_link[0])
#                     else:
#                         break  # avoid loop where chainee chains chainer
#                 except IndexError as e:
#                     pass
#         # find chainers that are not accounted for - these may be part of a circle...
#         flat_list = [item for ch in chain for item in ch[:-1]]  # flatten list excluding last in chain
#         leftovers = [x for x in flat_list if x not in chainer[:, frame_number]]  # count
#         if leftovers:
#             chain.append(leftovers)
#
#         this_chain_len = [len(x) for x in chain]
#         chain_length[:len(this_chain_len), frame_number] = this_chain_len
#
#         # which fly is in which chain? make boolean array [nflies, nchains] - allows to count flies twice for branching chains
#         for chain_count, chain_members in enumerate(chain):
#             chain_id[chain_members, chain_count, frame_number] = True
#
#     return chain_length, chain_id
#
#
# # better chain fun:
# # from edges (chainer-chainee)
# # find sub networks
# # for each sub-network
# #   find SIZE of network and who belongs to it
# def get_subnetworks(chainer, chainee, nflies):


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

    for frame_number in range(0, nframes):
        edges = [(int(source), int(target)) for source, target in zip(chainer[:,frame_number], chainee[:,frame_number]) if source>0]
        if len(edges):
            G = nx.Graph(edges)
            this_chains = sorted(nx.connected_components(G), key=len, reverse=True)

            # length of each chain
            this_chain_len = [len(this_chain) for this_chain in this_chains]
            chain_length[:len(this_chain_len), frame_number] = this_chain_len

            # which fly is in which chain? make boolean array [nflies, nchains] - allows to count flies twice for branching chains
            for this_chain_count, this_chain in enumerate(this_chains):
                try:
                    chain_id[list(this_chain), this_chain_count, frame_number] = True
                except:
                    import ipdb; ipdb.set_trace()

    return chain_length, chain_id
