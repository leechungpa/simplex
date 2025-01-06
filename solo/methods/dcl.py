from typing import Any, List, Sequence

import omegaconf
import torch
import torch.nn as nn
from solo.losses.dcl import dcl_loss_func
from solo.methods.base import BaseMethod
from solo.utils.misc import omegaconf_select
from solo.utils.eval_batch import evaluate_batch


# @evaluate_batch
class DCL(BaseMethod):
    def __init__(self, cfg: omegaconf.DictConfig):
        """Implements DCL (Debiased Contrastive Loss).

        Extra cfg settings:
            method_kwargs:
                proj_hidden_dim (int): number of neurons of the hidden layers of the projector.
                proj_output_dim (int): number of dimensions of projected features.
                tau (float): temperature parameter for the DCL loss.
        """
        super().__init__(cfg)

        self.tau: float = cfg.method_kwargs.tau

        proj_hidden_dim: int = cfg.method_kwargs.proj_hidden_dim
        proj_output_dim: int = cfg.method_kwargs.proj_output_dim

        # projector
        self.projector = nn.Sequential(
            nn.Linear(self.features_dim, proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_output_dim),
        )

    @staticmethod
    def add_and_assert_specific_cfg(cfg: omegaconf.DictConfig) -> omegaconf.DictConfig:
        """Adds method-specific default values/checks for config.

        Args:
            cfg (omegaconf.DictConfig): DictConfig object.

        Returns:
            omegaconf.DictConfig: same as the argument, used to avoid errors.
        """
        cfg = super(DCL, DCL).add_and_assert_specific_cfg(cfg)

        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.proj_hidden_dim")
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.proj_output_dim")

        cfg.method_kwargs.tau = omegaconf_select(cfg, "method_kwargs.tau", 0.1)

        return cfg

    @property
    def learnable_params(self) -> List[dict]:
        """Adds projector parameters to parent's learnable parameters.

        Returns:
            List[dict]: list of learnable parameters.
        """
        extra_learnable_params = [{"name": "projector", "params": self.projector.parameters()}]
        return super().learnable_params + extra_learnable_params

    def forward(self, X):
        """Performs the forward pass of the backbone and the projector.

        Args:
            X (torch.Tensor): a batch of images in the tensor format.

        Returns:
            Dict[str, Any]: a dict containing the outputs of the parent and the projected features.
        """
        out = super().forward(X)
        z = self.projector(out["feats"])
        out.update({"z": z})
        return out

    def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
        """Training step for DCL reusing BaseMethod training step.

        Args:
            batch (Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size num_crops containing batches of images.
            batch_idx (int): index of the batch.

        Returns:
            torch.Tensor: total loss composed of DCL loss and classification loss.
        """
        out = super().training_step(batch, batch_idx)
        class_loss = out["loss"]
        z1, z2 = out["z"]

        # ------- dcl loss -------
        dcl_loss = dcl_loss_func(z1, z2, tau=self.tau)

        self.log("train_loss", dcl_loss, on_epoch=True, sync_dist=True)

        return dcl_loss + class_loss
