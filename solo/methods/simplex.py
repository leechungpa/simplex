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

from typing import Any, List, Sequence

import omegaconf
import torch
import torch.nn as nn
from solo.losses.simplex import simplex_loss_func
from solo.methods.base import BaseMethod
from solo.utils.misc import omegaconf_select
from solo.utils.eval_batch import evaluate_batch

@evaluate_batch
class Simplex(BaseMethod):
    def __init__(self, cfg: omegaconf.DictConfig):
        """Implements Simplex

        Extra cfg settings:
            method_kwargs:
                proj_hidden_dim (int): number of neurons of the hidden layers of the projector.
                proj_output_dim (int): number of dimensions of projected features.
        """
        super().__init__(cfg)

        if isinstance(cfg.method_kwargs.k, (int, float)):
            self.parm_k: int = cfg.method_kwargs.k
        elif "num_instances" in cfg.data.keys():
            self.parm_k = cfg.data.num_instances
        else:
            raise ValueError("'method_kwargs.k' is needed.")
        print(f"Simplex loss k={self.parm_k}")

        self.parm_p: int = cfg.method_kwargs.p
        self.parm_lamb: int = cfg.method_kwargs.lamb

        self.rectify_large_neg_sim: bool = cfg.method_kwargs.rectify_large_neg_sim
        self.rectify_small_neg_sim: bool = cfg.method_kwargs.rectify_small_neg_sim

        self.unimodal: bool = cfg.method_kwargs.unimodal
        self.supervised_simplex: bool = cfg.method_kwargs.supervised_simplex

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
        """Adds method specific default values/checks for config.

        Args:
            cfg (omegaconf.DictConfig): DictConfig object.

        Returns:
            omegaconf.DictConfig: same as the argument, used to avoid errors.
        """

        cfg = super(Simplex, Simplex).add_and_assert_specific_cfg(cfg)

        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.proj_hidden_dim")
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.proj_output_dim")

        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.p")
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.lamb")

        cfg.method_kwargs.k = omegaconf_select(cfg, "method_kwargs.k", None)

        cfg.method_kwargs.rectify_large_neg_sim = omegaconf_select(cfg, "method_kwargs.rectify_large_neg_sim", False)
        cfg.method_kwargs.rectify_small_neg_sim = omegaconf_select(cfg, "method_kwargs.rectify_small_neg_sim", False)

        cfg.method_kwargs.unimodal = omegaconf_select(cfg, "method_kwargs.unimodal", True)
        cfg.method_kwargs.supervised_simplex = omegaconf_select(cfg, "method_kwargs.supervised_simplex", False)

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
        """Training step for Barlow Twins reusing BaseMethod training step.

        Args:
            batch (Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size num_crops containing batches of images.
            batch_idx (int): index of the batch.

        Returns:
            torch.Tensor: total loss composed of Barlow loss and classification loss.
        """ 
        if self.supervised_simplex:
            target = batch[-1]
        else:
            target = None

        out = super().training_step(batch, batch_idx)

        class_loss = out["loss"]
        z1, z2 = out["z"]

        # ------- simplex loss -------
        simplex_loss = simplex_loss_func(
            z1, z2, target=target,
            k=self.parm_k, p=self.parm_p, lamb=self.parm_lamb,
            rectify_large_neg_sim=self.rectify_large_neg_sim, rectify_small_neg_sim=self.rectify_small_neg_sim,
            unimodal=self.unimodal
        )

        self.log("train_loss", simplex_loss, on_epoch=True, sync_dist=True)

        return simplex_loss + class_loss
