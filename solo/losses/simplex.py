import torch
import torch.nn.functional as F
from solo.utils.misc import gather


def simplex_loss_func(
    z1: torch.Tensor, z2: torch.Tensor,
    target: torch.Tensor,
    k: int, p: int, lamb: float,
    rectify_large_neg_sim: bool = False, rectify_small_neg_sim: bool = False,
    unimodal: bool = True,
) -> torch.Tensor:
    """Computes Simplex loss given batch of projected features z1 from view 1 and
    projected features z2 from view 2.

    Args:
        z1 (torch.Tensor): NxD Tensor containing projected features from view 1.
        z2 (torch.Tensor): NxD Tensor containing projected features from view 2.
        k (int): See the definition. 
        p (int): See the definition.
        lamb (float): See the definition.
        rectify_large_neg_sim (bool, optional): Rectify the negative similarity to 0,
                                                    if it is larger than -1/(k-1).
        rectify_small_neg_sim (bool, optional): Rectify the negative similarity to 0,
                                                    if it is smaller than -1/(k-1).
        unimodal (bool): Calcuate the additional loss terms for unimodal CL.

    Returns:
        torch.Tensor: Simplex loss.
    """
    gathered_target = gather(target)

    target = target.unsqueeze(0)
    gathered_target = gathered_target.unsqueeze(0)

    pos_mask = target.t() == gathered_target

    neg_mask = ~ pos_mask

    # Calcuate the loss for both unimodal and bimodal CL
    z1 = F.normalize(z1, dim=1) # ( B, proj_output_dim )
    z2 = F.normalize(z2, dim=1) # ( B, proj_output_dim )

    batch_size = z1.size(0) # B

    gathered_z2 = gather(z2)
    similiarity = torch.einsum("id, jd -> ij", z1, gathered_z2)

    similiarity[pos_mask] = similiarity[pos_mask] - 1
    similiarity[neg_mask] = similiarity[neg_mask] + 1/(k-1)

    if rectify_large_neg_sim:
        # adjust to 0 if the similarity is greater than -1/(k-1)
        similiarity[neg_mask] = -F.relu(-similiarity[neg_mask])
    if rectify_small_neg_sim:
        # adjust to 0 if the similarity is simply less than -1/(k-1)
        similiarity[neg_mask] = F.relu(similiarity[neg_mask])

    similiarity = similiarity.abs().pow(p)

    loss = similiarity[pos_mask].sum() / batch_size + similiarity[neg_mask].sum() * lamb / (batch_size * (batch_size - 1))

    ############
    # To-Do: Remove the legacy.
    if unimodal:
        # Calcuate the additional loss terms for unimodal CL
        similiarity_z1 = torch.einsum("id, jd -> ij", z1, z1) # ( B, B )
        similiarity_z2 = torch.einsum("id, jd -> ij", z2, z2) # ( B, B )

        similiarity_z1[neg_mask] = similiarity_z1[neg_mask] + 1/(k-1)
        similiarity_z2[neg_mask] = similiarity_z2[neg_mask] + 1/(k-1)

        if rectify_large_neg_sim:
            # adjust to 0 if the similarity is greater than -1/(k-1)
            similiarity_z1[neg_mask] = -F.relu(-similiarity_z1[neg_mask])
            similiarity_z2[neg_mask] = -F.relu(-similiarity_z2[neg_mask])
        if rectify_small_neg_sim:
            # adjust to 0 if the similarity is simply less than -1/(k-1)
            similiarity_z1[neg_mask] = F.relu(similiarity_z1[neg_mask])
            similiarity_z2[neg_mask] = F.relu(similiarity_z2[neg_mask])

        similiarity_z1 = similiarity_z1.abs().pow(p)
        similiarity_z2 = similiarity_z2.abs().pow(p)

        loss = loss + (similiarity_z1[neg_mask].sum()*lamb/2 + similiarity_z2[neg_mask].sum()*lamb/2) / (batch_size * (batch_size - 1))
    ############

    return loss
