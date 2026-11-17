import numpy as np


def viterbi(scores, penalty):
    n, k = scores.shape
    back = np.zeros((n, k), dtype=np.int8)
    value = scores[0].astype(np.float64).copy()
    for t in range(1, n):
        candidate = value[:, None] - penalty
        best_from = candidate.argmax(axis=0)
        back[t] = best_from
        value = candidate[best_from, np.arange(k)] + scores[t]

    path = np.empty(n, dtype=np.int8)
    path[-1] = int(value.argmax())
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def viterbi2(evidence, cost_01, cost_10):
    values = np.asarray(evidence, dtype=np.float64).tolist()
    n = len(values)
    from_one_into_zero = bytearray(n)
    from_one_into_one = bytearray(n)
    in_zero, in_one = values[0], 0.0
    for t in range(1, n):
        switch = in_one - cost_10
        if switch > in_zero:
            from_one_into_zero[t] = 1
            new_zero = switch + values[t]
        else:
            new_zero = in_zero + values[t]
        switch = in_zero - cost_01
        if in_one > switch:
            from_one_into_one[t] = 1
            new_one = in_one
        else:
            new_one = switch
        in_zero, in_one = new_zero, new_one
    path = np.zeros(n, dtype=bool)
    state = 1 if in_one > in_zero else 0
    for t in range(n - 1, -1, -1):
        path[t] = state
        state = (from_one_into_one if state else from_one_into_zero)[t]
    return path


def runs_of(mask):
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def segments_of(codes, start=0):
    if not len(codes):
        return []
    breaks = np.flatnonzero(codes[1:] != codes[:-1]) + 1
    bounds = np.concatenate(([0], breaks, [len(codes)]))
    return [(start + int(bounds[i]), start + int(bounds[i + 1]), int(codes[bounds[i]]))
            for i in range(len(bounds) - 1)]
