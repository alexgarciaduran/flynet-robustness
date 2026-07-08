"""Structural and weighted recurrence of connectivity matrices."""

from collections import deque

import numpy as np
from scipy.sparse import csr_matrix, issparse
from scipy.sparse.csgraph import connected_components, dijkstra


def weight_matrix_from_network(network, signed=True):
    """Dense weight matrix W[i,j] from a flyvis network connectome."""
    params = network._param_api()
    nodes = network.connectome.nodes.to_df()
    edges = network.connectome.edges.to_df()
    W = np.zeros([len(nodes), len(nodes)], dtype=np.float32)
    if signed:
        w = params.edges.sign * params.edges.syn_count * params.edges.syn_strength
    else:
        w = params.edges.weight
    W[edges.source_index, edges.target_index] = w.detach().cpu().numpy()
    return W


def _bfs_distances(indptr, indices, source, n):
    dist = np.full(n, -1, dtype=np.int32)
    dist[source] = 0
    queue = deque([source])
    while queue:
        u = queue.popleft()
        for v in indices[indptr[u]:indptr[u + 1]]:
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                queue.append(v)
    return dist


def recurrence(W, n_samples=500, seed=0):
    """Mean shortest-cycle length over sampled nodes lying on a cycle.

    The shortest cycle through i is 1 + min dist(j->i) over successors j, found
    with one reverse BFS per sampled node. Returns (cycle_lengths, stats).
    """
    A = (W if issparse(W) else csr_matrix(W))
    A = (A != 0).astype(np.int8).tocsr()
    A.setdiag(0)
    A.eliminate_zeros()
    n = A.shape[0]
    AT = A.T.tocsr()
    rng = np.random.default_rng(seed)
    sample = rng.choice(n, size=min(n_samples, n), replace=False)

    cycle_lens = []
    for i in sample:
        succ = A.indices[A.indptr[i]:A.indptr[i + 1]]
        if len(succ) == 0:
            continue
        rev = _bfs_distances(AT.indptr, AT.indices, i, n)
        reachable = succ[rev[succ] >= 0]
        if len(reachable):
            cycle_lens.append(1 + int(rev[reachable].min()))

    nc = len(cycle_lens)
    stats = {"cycle_fraction": nc / len(sample), "n_cycles": nc,
             "mean_cycle_length": float(np.mean(cycle_lens)) if nc else np.inf,
             "median_cycle_length": float(np.median(cycle_lens)) if nc else np.inf}
    return cycle_lens, stats


def recurrence_weighted(W, n_samples=500, seed=0, eps=1e-12):
    """Effective recurrence: strongest cycle through each node by |W| product.

    Edge cost = -log|W|; per sample uses Dijkstra to find the max-product cycle.
    Returns mean_loggain, mean_gain, mean_hops, frac_on_cycle.
    """
    A = (W if issparse(W) else csr_matrix(W)).tocsr()
    n = A.shape[0]
    mag = A.copy()
    mag.data = -np.log(np.abs(mag.data) + eps)
    c = max(0.0, -mag.data.min()) + 1e-9
    mag_shifted = mag.copy()
    mag_shifted.data = mag.data + c

    rng = np.random.default_rng(seed)
    sample = rng.choice(n, size=min(n_samples, n), replace=False)
    loggains, hops = [], []
    for i in sample:
        succ = A.indices[A.indptr[i]:A.indptr[i + 1]]
        best_loggain, best_hops = np.inf, 0
        for j in succ:
            dist_s, preds = dijkstra(mag_shifted, directed=True, indices=j,
                                     return_predecessors=True)
            if not np.isfinite(dist_s[i]):
                continue
            path_hops, cur = 0, i
            while cur != j:
                cur = preds[cur]
                path_hops += 1
            jtoi = dist_s[i] - c * path_hops
            ij = -np.log(abs(A[i, j]) + eps)
            total = ij + jtoi
            if total < best_loggain:
                best_loggain, best_hops = total, path_hops + 1
        if np.isfinite(best_loggain):
            loggains.append(best_loggain)
            hops.append(best_hops)

    if not loggains:
        return dict(mean_loggain=np.inf, mean_gain=0.0, mean_hops=np.inf, frac_on_cycle=0.0)
    loggains = np.array(loggains)
    return dict(mean_loggain=float(loggains.mean()), mean_gain=float(np.exp(-loggains).mean()),
                mean_hops=float(np.mean(hops)), frac_on_cycle=len(loggains) / len(sample))


def recurrence_metrics(W):
    """Structural recurrence summary (self-loops, reciprocity, SCC structure)."""
    W = W if issparse(W) else csr_matrix(W)
    W_bin = (W != 0).astype(np.float32)
    n = W_bin.shape[0]
    reciprocal = W_bin.multiply(W_bin.T.tocsr())
    n_scc, labels = connected_components(W_bin, connection="strong")
    scc_sizes = np.bincount(labels)
    rows, cols = W_bin.nonzero()
    return {
        "self_loop_rate": float(np.mean(W_bin.diagonal() > 0)),
        "reciprocal_edge_rate": (reciprocal.nnz // 2) / W_bin.nnz,
        "recurrent_neuron_rate": float(np.mean(scc_sizes[labels] > 1)),
        "largest_scc": scc_sizes.max() / n,
        "recurrent_edge_rate": float(np.mean(labels[rows] == labels[cols])),
        "n_scc_normalized": n_scc / n,
    }
