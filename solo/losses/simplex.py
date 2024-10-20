# Copyright 2023 solo-learn development team.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import torch
import torch.nn.functional as F


def simplex_loss_func(
    z1: torch.Tensor, z2: torch.Tensor,
    k: int, p: int, lamb: float,
    rectify_large_neg_sim: bool = False, rectify_small_neg_sim: bool = False,
) -> torch.Tensor:
    """Computes Simplex loss given batch of projected features z1 from view 1 and
    projected features z2 from view 2.

    Args:
        z1 (torch.Tensor): NxD Tensor containing projected features from view 1.
        z2 (torch.Tensor): NxD Tensor containing projected features from view 2.
        k (int): See the definition. 
        p (int): See the definition.
        lamb (float): See the definition.
        rectify_large_neg_sim (bool, optional): Retify the negative similarity to 0,
                                                    if it is larger than -1/(k-1).
        rectify_small_neg_sim (bool, optional): Retify the negative similarity to 0,
                                                    if it is smaller than -1/(k-1).

    Returns:
        torch.Tensor: Simplex loss.
    """
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    similiarity = torch.einsum("id, jd -> ij", z1, z2)

    pos_mask = torch.eye(z1.size()[0], device=similiarity.device, dtype=torch.bool)
    neg_mask = ~ pos_mask

    similiarity[pos_mask] = similiarity[pos_mask] - 1
    similiarity[neg_mask] = similiarity[neg_mask] + 1/(k-1)

    if rectify_large_neg_sim:
        similiarity[neg_mask] = -F.relu(-similiarity[neg_mask])
    if rectify_small_neg_sim:
        similiarity[neg_mask] = F.relu(similiarity[neg_mask])

    similiarity = similiarity.abs().pow(p)

    loss = similiarity[pos_mask].mean() + similiarity[neg_mask].mean()*lamb

    return loss.sum()