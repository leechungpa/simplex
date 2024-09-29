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

import torch.distributed as dist


def simplex_loss_func(
    z1: torch.Tensor, z2: torch.Tensor, k: int, p: int
) -> torch.Tensor:
    """Computes Simplex' loss given batch of projected features z1 from view 1 and
    projected features z2 from view 2.

    Args:
        z1 (torch.Tensor): NxD Tensor containing projected features from view 1.
        z2 (torch.Tensor): NxD Tensor containing projected features from view 2.
        lamb (float, optional): off-diagonal scaling factor for the cross-covariance matrix.
            Defaults to 5e-3.
        scale_loss (float, optional): final scaling factor of the loss. Defaults to 0.025.

    Returns:
        torch.Tensor: Barlow Twins' loss.
    """

    N, _ = z1.size()

    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    similiarity = torch.einsum("id, jd -> ij", z1, z2)

    pos_mask = torch.eye(N, device=similiarity.device)
    neg_mask = torch.ones(N, N, device=similiarity.device) - pos_mask

    similiarity = (similiarity - pos_mask + (neg_mask/(k-1))).pow(p)
    loss = similiarity*pos_mask/N + similiarity*neg_mask/N/(N-1)

    return loss.sum()