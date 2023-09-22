from pathlib import Path
from logging import getLogger
from typing import Callable, Any, Dict, List, Union, Optional

import numpy as np
import torch
from torch import Tensor, nn
from pyserini.encode import AnceDocumentEncoder, AnceEncoder
from tqdm import tqdm
import torch.distributed as dist

logger = getLogger(__name__)

try:
    from tevatron.modeling.encoder import EncoderOutput
except ModuleNotFoundError as e:
    logger.warn("Failed to import tevatron ")
    logger.warn(
        "It seems that train env does not activated or python used is not for train env"
    )


def load_ance(device: str, **model_confs):
    return AnceTextEncoder(device=device, **model_confs)


class AnceTextEncoder:
    def __init__(
        self,
        model_path: str = "castorini/ance-msmarco-passage",
        tokenizer_name: Optional[str] = None,
        max_length: int = 512,
        device: str = "cuda:0",
    ) -> None:
        if not torch.cuda.is_available():
            logger.warn("cuda is not avaliable. AnceEncoder use cpu as device")
            device = "cpu"
        self.model_path = model_path
        self.device = device

        logger.info(f"AnceTextEncoder: Loading ANCE from {self.model_path}")
        self.encoder = AnceDocumentEncoder(
            model_name=self.model_path,
            tokenizer_name=(tokenizer_name or model_path),
            device=self.device,
        )
        self.encoder.model.eval()
        self.tokenizer = self.encoder.tokenizer
        self.max_length = max_length
        self.query_max_length = 64

    def to(self, device: str) -> None:
        self.encoder.device = device
        self.encoder.model.to(device)

    def encode(
        self,
        inputs: List[Dict[str, str]],
        batch_size: int = 16,
        device: Union[str, None] = None,
        convert_to_tensor: bool = True,
        verbose: bool = True,
    ) -> Union[torch.Tensor, np.ndarray]:
        device = self.device if device is None else device
        self.to(device)

        inputs_len = len(inputs)
        embs = []

        offset_iter = (
            tqdm(range(0, inputs_len, batch_size), total=(inputs_len // batch_size))
            if verbose
            else range(0, inputs_len, batch_size)
        )
        for i in offset_iter:
            batch_inputs = inputs[i : min(i + batch_size, inputs_len)]
            batch_texts, batch_titles = [], []
            for input_ in batch_inputs:
                batch_texts.append(input_["text"])
                if "title" in input_:
                    batch_titles.append(input_["title"])
            kwargs = {
                "texts": batch_texts,
                "titles": None if len(batch_titles) == 0 else batch_titles,
                "fp16": False,
                "max_length": self.max_length,
                "add_sep": False,
            }
            embedding = self.encoder.encode(**kwargs)
            embs.append(embedding)
        embeddings: np.ndarray = np.concatenate(embs)
        if convert_to_tensor:
            return torch.from_numpy(embeddings)
        return embeddings

    def encode_queries(
        self,
        queries: List[str],
        batch_size: int,
        convert_to_tensor: bool = True,
        **kwargs: Dict[str, Any],
    ) -> Union[torch.Tensor, np.ndarray]:
        dict_queries = [{"text": query} for query in queries]
        query_embs = self.encode(
            dict_queries,
            batch_size=batch_size,
            convert_to_tensor=convert_to_tensor,
            **kwargs,
        )
        return query_embs

    def encode_corpus(
        self,
        corpus: List[Dict[str, str]],
        batch_size: int,
        convert_to_tensor: bool = True,
        **kwargs: Dict[str, Any],
    ) -> Union[torch.Tensor, np.ndarray]:
        corpus_emb = self.encode(
            corpus, batch_size=batch_size, convert_to_tensor=convert_to_tensor, **kwargs
        )
        return corpus_emb


class AnceForTrain(AnceEncoder):
    def __init__(self, config, negatives_x_device: bool = True, **kwargs):
        super().__init__(config)
        self.training = True
        self.cross_entropy = torch.nn.functional.cross_entropy
        # self.margin_ranking_loss = torch.nn.MarginRankingLoss()
        self.mse_loss = torch.nn.MSELoss()

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

        # loss = self.mse_loss(pos_scores, perturbed_pos_scores)
        y = -1 * torch.ones(len(pos_scores), device=scores.device)  # (batch,)
        loss = self.margin_ranking_loss(pos_scores, perturbed_pos_scores, y)

        return loss

    def compute_perturb_loss_mse_if_less(self, scores, target, perturbed_index):
        """
        Add loss when perturbed positive scores less than positive scores
        """
        # row_index is used for selecting positive doc score and perturbed score
        row_index = torch.arange(len(target))
        pos_scores = scores[row_index, target]  # (batch,)
        perturbed_pos_scores = scores[row_index, perturbed_index]  # (batch,)

        scores_mask = (pos_scores - perturbed_pos_scores) > 0.1
        pos_scores = torch.masked_select(pos_scores, scores_mask)
        perturbed_pos_scores = torch.masked_select(perturbed_pos_scores, scores_mask)

        if len(perturbed_pos_scores) <= 0:
            return 0

        loss = self.mse_loss(pos_scores, perturbed_pos_scores)

        return loss

    def compute_loss(self, scores, target, perturbed_index):
        contrastive_loss = self.compute_contrastive_loss(
            scores, target, perturbed_index
        )
        # perturbed_loss = self.compute_perturb_loss(scores, target, perturbed_index)
        perturbed_loss = self.compute_perturb_loss_mse_if_less(
            scores, target, perturbed_index
        )

        print(
            f"contrastive_loss, perturbed_loss = ({contrastive_loss}, {perturbed_loss})"
        )
        return contrastive_loss + (perturbed_loss)

    def encode(self, inputs):
        return super(AnceForTrain, self).forward(**inputs)

    def save(self, output_dir: str) -> None:
        self.save_pretrained(output_dir)

    def forward(
        self,
        query: Optional[Dict[str, torch.Tensor]] = None,
        passage: Optional[Dict[str, torch.Tensor]] = None,
    ):
        q_reps, p_reps = None, None
        if query is not None:
            q_reps = self.encode(query)
        if passage is not None:
            p_reps = self.encode(passage)

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
