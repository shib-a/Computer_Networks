"""
FedAvg aggregation.

Takes a list of (state_dict, num_samples) pairs and returns a
weighted-average state_dict. All arithmetic is done in float32 to
prevent integer tensor overflow; the result is cast back to the
original dtype of each parameter.
"""
import copy
from collections import OrderedDict
from typing import List, Tuple

import torch


def fedavg(updates: List[Tuple[OrderedDict, int]]) -> OrderedDict:
    """
    Weighted average of model state dicts.

    Args:
        updates: list of (state_dict, num_local_samples) from each client.

    Returns:
        Averaged state_dict with the same keys and dtypes as the inputs.
    """
    if not updates:
        raise ValueError("fedavg called with empty updates list")

    total = sum(n for _, n in updates)
    ref_sd, _ = updates[0]

    result = OrderedDict()
    for key in ref_sd.keys():
        original_dtype = ref_sd[key].dtype
        # weighted sum in float32
        weighted = sum(
            sd[key].float() * (n / total)
            for sd, n in updates
        )
        result[key] = weighted.to(original_dtype)

    return result
