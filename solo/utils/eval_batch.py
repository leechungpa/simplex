import torch
import torch.nn.functional as F


def evaluate_batch(cls):
    cls.__init__ = _evaluate_batch_init(cls.__init__)
    cls.training_step = _evaluate_batch(cls.training_step)

    return cls


def _evaluate_batch_init(func):
    def wrapper(self, *args, **kargs):
        init_outputs = func(self, *args, **kargs)
        if self.evaluate_batch.enable:
            self.automatic_optimization = False
            print("[Evaluate batch] 'automatic_optimization' for LightningModule sets to False.")
        return init_outputs
    return wrapper


def _evaluate_batch(func):
    def wrapper(self, batch, *args, **kargs):
        if self.evaluate_batch.enable:
            optimizer = self.optimizers()

            # ------- negative pair similarity (before optimization) -------
            with torch.no_grad():
                _, X, _ = batch

                z1 = self.projector(self.backbone(X[0]))
                z2 = self.projector(self.backbone(X[1]))

                uniformity_before = evaluate_avg_neg_similarity(z1, z2)
                self.log("uniformity_before", uniformity_before, on_epoch=True, sync_dist=True)
            
            # ------- forward loss -------
            loss = func(self, batch, *args, **kargs)

            # ------- backward loss -------
            optimizer.zero_grad()
            self.manual_backward(loss)
            optimizer.step()

            # ------- negative pair similarity (after optimization) -------
            with torch.no_grad():
                z1 = self.projector(self.backbone(X[0]))
                z2 = self.projector(self.backbone(X[1]))

                uniformity_after = evaluate_avg_neg_similarity(z1, z2)
                self.log("uniformity_after", uniformity_after, on_epoch=True, sync_dist=True)
                self.log("uniformity_diff", uniformity_after - uniformity_before, on_epoch=True, sync_dist=True)
            return loss
        else:
            return func(self, batch, *args, **kargs)
    return wrapper


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