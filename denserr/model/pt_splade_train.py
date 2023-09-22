from __future__ import annotations

from typing import Iterable, List, Any, Optional, Dict, Tuple
from collections import defaultdict
from tqdm import tqdm
import re
import os
import logging
from pathlib import Path

from denserr.utils.util import project_dir

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

try:
    from splade.models.transformer_rep import Splade
except ModuleNotFoundError as e:
    logger.warning("Failed to import Splade from splade module ")
    logger.warning("Make sure splade env is activated or using python for ptsplade env")

try:
    from tevatron.modeling.encoder import EncoderOutput
except ModuleNotFoundError as e:
    logger.warn("Failed to import tevatron ")
    logger.warn(
        "It seems that train env does not activated or python used is not for train env"
    )


class SpladeForTrain(Splade):
    def __init__(self, *args, negatives_x_device: bool = True, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.training = True
        self.cross_entropy = torch.nn.functional.cross_entropy
        self.margin_ranking_loss = torch.nn.MarginRankingLoss()

        self.negatives_x_device = negatives_x_device
        if self.negatives_x_device:
            if not dist.is_initialized():
                raise ValueError(
                    "Distributed training has not been initialized for representation all gather."
                )
            self.process_rank = dist.get_rank()
            self.world_size = dist.get_world_size()

    def compute_similarity(self, q_reps, p_reps):
        return torch.matmul(q_reps, p_reps.transpose(0, 1))

    def decompose_scores_with_perturbed(self, scores, perturbed_index):
        batch_size, num_passages = scores.shape
        mask = torch.nn.functional.one_hot(perturbed_index, num_classes=num_passages)
        mask = mask <= 0
        # (len(target) * num_passages,)
        scores = torch.masked_select(scores, mask)
        # (len(target), num_passages - 1) since removed 1 element from each dim=1
        scores = scores.view(-1, num_passages - 1)

        return scores

    def make_target(self, scores_size, device):
        target = torch.arange(scores_size[0], device=device, dtype=torch.long)
        target = target * (scores_size[1] // scores_size[0])
        return target

    def compute_contrastive_loss(self, scores, target, perturbed_index):
        scores = self.decompose_scores_with_perturbed(scores, perturbed_index)
        # target = self.make_target(scores.size(), scores.device)

        return self.cross_entropy(scores, target)

    def compute_perturb_loss(self, scores, target, perturbed_index):
        """
        Add loss when perturbed positive scores less than positive scores
        """
        # row_index is used for selecting positive doc score and perturbed score
        row_index = torch.arange(len(target))
        pos_scores = scores[row_index, target]  # (batch,)
        perturbed_pos_scores = scores[row_index, perturbed_index]  # (batch,)

        # loss = self.mse_loss(pos_scores, perturbed_pos_scores)
        y = -1 * torch.ones(len(pos_scores), device=scores.device)  # (batch,)
        loss = self.margin_ranking_loss(pos_scores, perturbed_pos_scores, y)

        return loss

    def compute_loss(self, scores, target, perturbed_index):
        contrastive_loss = self.compute_contrastive_loss(
            scores, target, perturbed_index
        )
        perturbed_loss = self.compute_perturb_loss(scores, target, perturbed_index)

        print(
            f"contrastive_loss, perturbed_loss = ({contrastive_loss}, {perturbed_loss})"
        )
        return contrastive_loss + (perturbed_loss)

    def splade_forward(self, kwargs):
        return super(SpladeForTrain, self).forward(**kwargs)

    def forward(
        self,
        query: Optional[Dict[str, torch.Tensor]] = None,
        passage: Optional[Dict[str, torch.Tensor]] = None,
    ):
        kwargs = {"q_kwargs": query, "d_kwargs": passage, "score_batch": True}
        q_reps, p_reps = None, None
        if query is not None:
            q_reps = self.splade_forward({"q_kwargs": query})["q_rep"]
        if passage is not None:
            p_reps = self.splade_forward({"d_kwargs": passage})["d_rep"]

        # for inference
        if q_reps is None or p_reps is None:
            return EncoderOutput(q_reps=q_reps, p_reps=p_reps)

        # for training
        if self.training:
            if self.negatives_x_device:
                q_reps = self._dist_gather_tensor(q_reps)
                p_reps = self._dist_gather_tensor(p_reps)
            scores = self.compute_similarity(q_reps, p_reps)
            scores = scores.view(q_reps.size(0), -1)

            target = self.make_target(scores.size(), scores.device)
            perturbed_index = target + 1  # for perturbed positive

            loss = self.compute_loss(scores, target, perturbed_index)
            # loss = self.compute_contrastive_loss(scores, perturbed_index)
            if self.negatives_x_device:
                loss = loss * self.world_size  # counter average weight reduction
        # for eval
        else:
            scores = self.compute_similarity(q_reps, p_reps)
            loss = None
        return EncoderOutput(
            loss=loss,
            scores=scores,
            q_reps=q_reps,
            p_reps=p_reps,
        )

    def save(self, output_dir: str, **kwargs) -> None:
        # it is practical (although redundant) to save model weights using huggingface API, because if the model has
        # no other params, we can reload it easily with .from_pretrained()
        self.transformer_rep.transformer.save_pretrained(output_dir)
        tokenizer = self.transformer_rep.tokenizer
        tokenizer.save_pretrained(output_dir)
        if self.transformer_rep_q is not None:
            output_dir_q = os.path.join(output_dir, "model_q")
            self.transformer_rep_q.transformer.save_pretrained(output_dir_q)
            tokenizer = self.transformer_rep_q.tokenizer
            tokenizer.save_pretrained(output_dir_q)

    def _dist_gather_tensor(self, t: Optional[torch.Tensor]):
        if t is None:
            return None
        t = t.contiguous()

        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)

        all_tensors[self.process_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)

        return all_tensors
