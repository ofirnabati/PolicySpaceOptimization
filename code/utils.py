# Code in this file is copied and adapted from
# https://github.com/openai/evolution-strategies-starter.

import numpy as np
import torch


def itergroups(items, group_size):
    assert group_size >= 1
    group = []
    for x in items:
        group.append(x)
        if len(group) == group_size:
            yield tuple(group)
            del group[:]
    if group:
        yield tuple(group)



def batched_weighted_sum(weights, vecs, batch_size):
    W1, b1 = vecs[0]
    N = len(vecs)
    W = [torch.zeros_like(w) for w in W1]
    bias = [torch.zeros_like(b) for b in b1]
    for id, vec in enumerate(vecs):
        W1, bias1 = vec
        W = [w + weights[id] / N for w,w1 in zip(W,W1)]
        bias = [b + weights[id] / N for b,b1 in zip(bias,bias1)]
    return [W, bias]

    # total = 0
    # num_items_summed = 0
    # for batch_weights, batch_vecs in zip(itergroups(weights, batch_size),
    #                                      itergroups(vecs, batch_size)):
    #     assert len(batch_weights) == len(batch_vecs) <= batch_size
    #     total += np.dot(np.asarray(batch_weights, dtype=np.float64),
    #                     np.asarray(batch_vecs, dtype=np.float64))
    #     num_items_summed += len(batch_weights)
    # return total, num_items_summed


class DictListObject(dict):
    """A dictionnary of lists of same size. Dictionnary items can be
    accessed using `.` notation and list items using `[]` notation.
    Example:
        >>> d = DictListObject({"a": [[1, 2], [3, 4]], "b": [[5], [6]]})
        >>> d.a
        [[1, 2], [3, 4]]
        >>> d[0]
        DictList({"a": [1, 2], "b": [5]})
    """

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    def __len__(self):
        return len(next(iter(dict.values(self))))

    def __getitem__(self, index):
        return DictListObject({key: value[index] for key, value in dict.items(self)})

    def __setitem__(self, index, d):
        for key, value in d.items():
            dict.__getitem__(self, key)[index] = value
