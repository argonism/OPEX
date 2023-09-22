import os
from typing import Any, Dict, Iterable, List, Optional
from logging import getLogger
from collections import defaultdict
from pathlib import Path
from dataclasses import dataclass

import pandas as pd
from colbert.modeling.colbert import ColBERT
from pyterrier_colbert import load_checkpoint
import torch
from torch import Tensor, nn
import torch.distributed as dist

import colbert.evaluation.loaders
from colbert.modeling.tokenization import QueryTokenizer, DocTokenizer
from colbert.utils.utils import save_checkpoint

from denserr.model._base import Retriever
from denserr.utils.util import index_base_path

logger = getLogger(__name__)

try:
    from tevatron.modeling.encoder import EncoderOutput
except ModuleNotFoundError as e:
    logger.warn("Failed to import tevatron ")
    logger.warn(
        "It seems that train env does not activated or python used is not for train env"
    )


@dataclass
class LoadColbertArgs:
    checkpoint: str = "http://www.dcs.gla.ac.uk/~craigm/ecir2021-tutorial/colbert_model_checkpoint.zip"
    query_maxlen: int = 32
    doc_maxlen: int = 180
    dim: int = 128
    similarity: str = "cosine"
    mask_punctuation: bool = False


def load_train_colbert():
    args = LoadColbertArgs()
    colbert = ColbertForTrain.from_pretrained(
        "bert-base-uncased",
        query_maxlen=args.query_maxlen,
        doc_maxlen=args.doc_maxlen,
        dim=args.dim,
        similarity_metric=args.similarity,
        mask_punctuation=args.mask_punctuation,
    )
    checkpoint = load_checkpoint(args.checkpoint, colbert, do_print=True)

    colbert.train()

    query_tokenizer = QueryTokenizer(colbert.query_maxlen)
    doc_tokenizer = DocTokenizer(colbert.doc_maxlen)

    return colbert, checkpoint, query_tokenizer, doc_tokenizer


class ColbertForTrain(ColBERT):
    def __init__(self, *args, negatives_x_device: bool = True, **kwargs):
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

    def decompose_scores(self, scores, perturbed_index):
        batch_size, num_passages = scores.shape
        # mask = torch.nn.functional.one_hot(perturbed_index, num_classes=num_passages)
        ones = torch.ones_like(scores)
        ones[:, perturbed_index] = 0
        mask = ones >= 1
        # (len(target) * num_passages,)
        scores = torch.masked_select(scores, mask)
        # (len(target), num_passages - 1) since removed 1 element from each dim=1
        scores = scores.view(-1, num_passages - batch_size)

        return scores

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

    def save(self, output_dir: str) -> None:
        output_path = os.path.join(output_dir, "colbert.dnn")
        checkpoint = {}
        checkpoint["model_state_dict"] = self.state_dict()
        torch.save(checkpoint, output_path)

    def compute_similarity(self, q_reps, p_reps):
        token_scores = torch.einsum("qin,pjn->qipj", q_reps, p_reps)
        scores, _ = token_scores.max(-1)
        scores = scores.sum(1)
        return scores

    def forward(
        self,
        query: Optional[Dict[str, torch.Tensor]] = None,
        passage: Optional[Dict[str, torch.Tensor]] = None,
    ):
        q_reps, p_reps = None, None
        if query is not None:
            q_reps = self.query(**query)
        if passage is not None:
            p_reps = self.doc(**passage)

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

    def _dist_gather_tensor(self, t: Optional[torch.Tensor]):
        if t is None:
            return None
        t = t.contiguous()

        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)

        all_tensors[self.process_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)

        return all_tensors
