import torch
import torch.nn.functional as F


def evaluate_avg_neg_similarity(z1, z2):
    """Evaluate the average of (cosine) similarity of negative pairs.

    Args:
        z1: (N, D)
        z2: (N, D)
    """
    batch_size = z1.size(0)
    
    # z1.unsqueeze(1): (N, 1, D)
    # z2.unsqueeze(0): (1, N, D)
    similarity_matrix = F.cosine_similarity(z1.unsqueeze(1), z2.unsqueeze(0), dim = 2)  # (N, N)

    mask = torch.eye(batch_size, dtype=torch.bool)  # (N, N)

    # similarity_matrix[~mask]: ( N*(N-1), )
    avg_neg_similarity = similarity_matrix[~mask].mean()

    return avg_neg_similarity