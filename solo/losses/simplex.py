import torch
import torch.nn.functional as F
from solo.utils.misc import gather

def lamb_scheduler(initial_lamb: float, decay_rate: float, epoch: int, step_size: int = 1) -> float:
    """
    Args:
        initial_lamb (float): Initial value of lambda.
        decay_rate (float): Decay rate for lambda.
        epoch (int): Current epoch.
        step_size (int): Number of epochs before applying the decay.
    """
    if epoch % step_size == 0:
        return initial_lamb * (decay_rate ** (epoch // step_size))
    return initial_lamb * (decay_rate ** ((epoch - (epoch % step_size)) // step_size))



def simplex_loss_func_general(
    z1: torch.Tensor, z2: torch.Tensor,
    target: torch.Tensor,
    k: int, p: int, lamb: float,
    centroid: torch.Tensor|None = None,
    rectify_large_neg_sim: bool = False, rectify_small_neg_sim: bool = False,
    disable_positive_term: bool = False,
) -> torch.Tensor:
    gathered_target = gather(target)

    target = target.unsqueeze(0)
    gathered_target = gathered_target.unsqueeze(0)

    pos_mask = target.t() == gathered_target

    neg_mask = ~ pos_mask

    z1 = F.normalize(z1, dim=1) # ( B, proj_output_dim )
    z2 = F.normalize(z2, dim=1) # ( B, proj_output_dim )

    if centroid is not None:
        z1 = z1 - centroid
        z2 = z2 - centroid
        scale = 1 - torch.dot(centroid.t(), centroid)
    else:
        scale = 1

    gathered_z2 = gather(z2)
    similarity = torch.einsum("id, jd -> ij", z1, gathered_z2)

    if not disable_positive_term:
        similarity[pos_mask] = similarity[pos_mask] - 1 * scale
    # similarity[pos_mask] = similarity[pos_mask] - 0.743632495403289
    similarity[neg_mask] = similarity[neg_mask] + 1/(k-1) * scale
    # similarity[neg_mask] = similarity[neg_mask] - 0.0233505647629499

    if rectify_large_neg_sim:
        # adjust to 0 if the similarity is greater than -1/(k-1)
        similarity[neg_mask] = -F.relu(-similarity[neg_mask])
    if rectify_small_neg_sim:
        # adjust to 0 if the similarity is simply less than -1/(k-1)
        similarity[neg_mask] = F.relu(similarity[neg_mask])

    similarity = similarity.abs().pow(p)

    loss = 0.0
    if not disable_positive_term:
        loss += similarity[pos_mask].mean()
    loss += similarity[neg_mask].mean() * lamb
    return loss


def simplex_loss_func(
    z1: torch.Tensor, z2: torch.Tensor,
    target: torch.Tensor,
    p: int, lamb: float,
    delta: float = None, k: int = None,
    rectify_large_neg_sim: bool = False, rectify_small_neg_sim: bool = False,
    # unimodal: bool = True,
    disable_positive_term: bool = False,
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
    use_negative_from_same_branch = False

        # Calcuate the loss for both unimodal and bimodal CL
    z1 = F.normalize(z1, dim=1) # ( B, proj_output_dim )
    z2 = F.normalize(z2, dim=1) # ( B, proj_output_dim )

    gathered_z2 = gather(z2)
    gathered_target = gather(target)

    similarity = torch.einsum("id, jd -> ij", z1, gathered_z2)

    target = target.unsqueeze(0)
    gathered_target = gathered_target.unsqueeze(0)
    
    pos_mask = target.t() == gathered_target
    pos_mask.fill_diagonal_(0)

    neg_mask = target.t() != gathered_target

    if delta is None and k is None:
        raise ValueError("Either `delta` or `k` must be provided.")
    if delta is not None and k is not None:
        raise ValueError("Provide only one of `delta` or `k`, not both.")
    

    if delta is None:
        delta = - 1/(k-1)

    similarity[pos_mask] = similarity[pos_mask] - 1

    similarity[neg_mask] = similarity[neg_mask] - delta

    similarity = similarity.abs().pow(p)

    if use_negative_from_same_branch:
        similarity_z1 = torch.einsum("id, jd -> ij", z1, z1)
        similarity_z2 = torch.einsum("id, jd -> ij", z2, z2)

        similarity_z1[neg_mask] = similarity_z1[neg_mask] - delta
        similarity_z2[neg_mask] = similarity_z2[neg_mask] - delta

        similarity_z1 = similarity_z1.abs().pow(p)
        similarity_z2 = similarity_z2.abs().pow(p)


    if not use_negative_from_same_branch:
        loss = similarity[neg_mask].mean() * lamb
    else:
        loss = (similarity_z1[neg_mask].mean() + similarity_z2[neg_mask].mean()) / 2 * lamb

    if not disable_positive_term:
        loss += similarity[pos_mask].mean()

    return loss