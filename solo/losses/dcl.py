import torch
import torch.nn.functional as F
from solo.utils.misc import gather, get_rank

def dcl_loss_func(
    z: torch.Tensor, indexes: torch.Tensor, temperature: float = 0.1
) -> torch.Tensor:
    """
    Args:
        z: (N*views) x D Tensor containing projected features from the views.
        temperature (float): Temperature scaling factor.

    Returns:
        torch.Tensor: DCL loss.
    """

    z = F.normalize(z, dim=-1)
    gathered_z = gather(z)
    sim = torch.exp(torch.einsum("if, jf -> ij", z, gathered_z) / temperature)
    gathered_indexes = gather(indexes)
    indexes = indexes.unsqueeze(0)
    gathered_indexes = gathered_indexes.unsqueeze(0)

    # negatives
    pos_mask = indexes.t() == gathered_indexes
    neg_mask = indexes.t() != gathered_indexes
    neg_mask[:, z.size(0) * get_rank() :].fill_diagonal_(0)
    pos = torch.sum(sim * pos_mask, dim=1)
    neg = torch.sum(sim * neg_mask, 1)
    loss = -torch.mean(torch.log(pos / (pos + neg)))
    return loss