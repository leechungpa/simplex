import torch
import torch.nn.functional as F
from solo.utils.misc import gather

def dcl_loss_func(
    z1: torch.Tensor, z2: torch.Tensor, tau: float = 0.1
) -> torch.Tensor:
    """
    Computes Debiased Contrastive Loss (DCL) for a batch of projected features.

    Args:
        z1 (torch.Tensor): NxD Tensor containing projected features from view 1.
        z2 (torch.Tensor): NxD Tensor containing projected features from view 2.
        tau (float): Temperature scaling factor (default: 0.1).

    Returns:
        torch.Tensor: DCL loss.
    """

    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)

    gathered_z1 = gather(z1)
    gathered_z2 = gather(z2)

    sim_11 = torch.exp(torch.mm(z1, gathered_z1.t()) / tau)  # z1 vs z1
    sim_22 = torch.exp(torch.mm(z2, gathered_z2.t()) / tau)  # z2 vs z2
    sim_12 = torch.exp(torch.mm(z1, gathered_z2.t()) / tau)  # z1 vs z2
    sim_21 = torch.exp(torch.mm(z2, gathered_z1.t()) / tau)  # z2 vs z1

    batch_size = z1.size(0)
    rank_offset = batch_size * (torch.distributed.get_rank() if torch.distributed.is_initialized() else 0)
    mask = torch.eye(batch_size, device=z1.device).bool()
    pos_mask_12 = mask.repeat(1, gathered_z2.size(0) // batch_size)
    pos_mask_21 = mask.repeat(1, gathered_z1.size(0) // batch_size)

    pos_12 = sim_12[pos_mask_12].view(batch_size, -1)
    pos_21 = sim_21[pos_mask_21].view(batch_size, -1)

    neg_12 = torch.sum(sim_12, dim=1) - pos_12.sum(dim=1)
    neg_21 = torch.sum(sim_21, dim=1) - pos_21.sum(dim=1)
    neg_11 = torch.sum(sim_11.masked_fill(mask, 0), dim=1)
    neg_22 = torch.sum(sim_22.masked_fill(mask, 0), dim=1)

    loss_12 = -torch.log(pos_12.sum(dim=1) / (pos_12.sum(dim=1) + neg_12 + neg_11))
    loss_21 = -torch.log(pos_21.sum(dim=1) / (pos_21.sum(dim=1) + neg_21 + neg_22))
    loss = (loss_12 + loss_21).mean()

    return loss