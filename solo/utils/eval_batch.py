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
            sch = self.lr_schedulers()

            _, X, _ = batch

            # ------- before optimization -------
            if not self.evaluate_batch.skip_before_optm:
                with torch.no_grad():
                    z1 = F.normalize(self.projector(self.backbone(X[0]))) # ( B, proj_output_dim )
                    z2 = F.normalize(self.projector(self.backbone(X[1]))) # ( B, proj_output_dim )

                    centroid_z1 = z1.mean(0)
                    centroid_z2 = z2.mean(0)
                    centroid = (centroid_z1+centroid_z2) / 2

                    pos_sim, neg_sim = eval_similarity(z1, z2)

                    # TBD: if self.evaluate_batch.type == "all":
                    result_before = {
                        "alignment": eval_alignment(z1, z2),
                        "uniformity": (eval_uniformity(z1)+eval_uniformity(z2)) / 2,
                        "pos_sim": pos_sim,
                        "neg_sim": neg_sim,
                        "centroid": centroid.norm(p=2).item(),
                        "centroid_z1": centroid_z1.norm(p=2).item(),
                        "centroid_z2": centroid_z2.norm(p=2).item(),
                    }
                    for key, value in result_before.items():
                        self.log("[before] "+key, value, on_epoch=True, sync_dist=True)

                    del z1, z2

            # ------- forward and backward -------
            loss = func(self, batch, *args, **kargs)

            optimizer.zero_grad()
            self.manual_backward(loss)
            optimizer.step()
            sch.step()

            # ------- after optimization -------
            with torch.no_grad():
                z1 = F.normalize(self.projector(self.backbone(X[0]))) # ( B, proj_output_dim )
                z2 = F.normalize(self.projector(self.backbone(X[1]))) # ( B, proj_output_dim )

                centroid_z1 = z1.mean(0)
                centroid_z2 = z2.mean(0)
                centroid = (centroid_z1+centroid_z2) / 2

                pos_sim, pos_sim_std, neg_sim, neg_sim_std = eval_similarity(z1, z2, show_std=True)

                # TBD: if self.evaluate_batch.type == "all":
                result_after = {
                    "alignment": eval_alignment(z1, z2),
                    "uniformity": (eval_uniformity(z1)+eval_uniformity(z2)) / 2,
                    "pos_sim": pos_sim,
                    "neg_sim": neg_sim,
                    "centroid": centroid.norm(p=2).item(),
                    "centroid_z1": centroid_z1.norm(p=2).item(),
                    "centroid_z2": centroid_z2.norm(p=2).item(),
                }
                for key, value in result_after.items():
                    self.log("[after] "+key, value, on_epoch=True, sync_dist=True)
                    if not self.evaluate_batch.skip_before_optm:
                        self.log("[diff] "+key, value - result_before[key], on_epoch=True, sync_dist=True)
                    
                self.log("[after] pos_sim_std", pos_sim_std, on_epoch=True, sync_dist=True)
                self.log("[after] neg_sim_std", neg_sim_std, on_epoch=True, sync_dist=True)

                del z1, z2

            return loss
        else:
            return func(self, batch, *args, **kargs)
    return wrapper


def eval_similarity(z1, z2, show_std=False):
    """Evaluate the average of (cosine) similarity of positive / negative pairs.

    Args:
        z1: (N, D)
        z2: (N, D)
    """
    # z1.unsqueeze(1): (N, 1, D)
    # z2.unsqueeze(0): (1, N, D)
    similarity_matrix = F.cosine_similarity(z1.unsqueeze(1), z2.unsqueeze(0), dim = 2)  # (N, N)

    mask = torch.eye(z1.size(0), dtype=torch.bool)  # (N, N)

    if show_std:
        return similarity_matrix[mask].mean(), similarity_matrix[mask].std(), similarity_matrix[~mask].mean(), similarity_matrix[~mask].std()    
    else:
        # similarity of positive pairs, similarity of negative pairs
        return similarity_matrix[mask].mean(), similarity_matrix[~mask].mean()

# https://github.com/ssnl/align_uniform
def eval_alignment(x, y, alpha=2):
    return (x - y).norm(p=2, dim=1).pow(alpha).mean()

def eval_uniformity(x, t=2):
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()