import torch
import torch.nn.functional as F
from solo.utils.misc import gather



def add_vrn_loss_term(
    z1: torch.Tensor, z2: torch.Tensor,
    target: torch.Tensor,
    p: int, lamb: float,
    delta: float = None, k: int = None,
) -> torch.Tensor:
    """
    Computes the VRN (Variance-Reduction for Negative-pair similarity) loss term.
    
    Args:
        z1 (torch.Tensor): Normalized embedding vectors with shape (B, proj_output_dim).
        z2 (torch.Tensor): Normalized embedding vectors with shape (B, proj_output_dim).
        target (torch.Tensor): Target labels in the batch.
        p (int): Exponent value for the power operation in the loss formulation.
        lamb (float): Weighting factor for the loss term.
        delta (float, optional): A predefined similarity threshold for negative pairs. Default is None.
        k (int, optional): The number of negative pairs. Default is None.
    
    Returns:
        torch.Tensor: The computed VRN loss value.
    """

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

    loss = similarity[neg_mask].mean() * lamb

    return loss