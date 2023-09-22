from logging import getLogger
from typing import Any, Dict, List, Union

import numpy as np
import torch
from pyserini.encode import DprDocumentEncoder, DprQueryEncoder
from tqdm import tqdm

logger = getLogger(__name__)


class DPRTextEncoder:
    def __init__(
        self,
        max_length: int = 256,
        ctx_device: str = "cuda:0",
        query_device: str = "cuda:0",
    ) -> None:
        if not torch.cuda.is_available():
            logger.warn("cuda is not avaliable. AnceEncoder use cpu as device")
            ctx_device, query_device = "cpu", "cpu"
        self.doc_encoder = DprDocumentEncoder(
            model_name="facebook/dpr-ctx_encoder-multiset-base", device=ctx_device
        )
        self.query_encoder = DprQueryEncoder(
            model_name="facebook/dpr-question_encoder-multiset-base",
            device=query_device,
        )
        self.doc_encoder.model.eval()
        self.query_encoder.model.eval()
        self.tokenizer = self.doc_encoder.tokenizer
        self.max_length = max_length
        self.query_max_length = 64

    def to(self, ctx_device: str, query_device: str) -> None:
        self.doc_encoder.device = ctx_device
        self.doc_encoder.model.to(ctx_device)
        self.query_encoder.device = query_device
        self.query_encoder.model.to(query_device)

    def encode(
        self,
        inputs: List[Dict[str, str]],
        is_query: bool = False,
        batch_size: int = 16,
        convert_to_tensor: bool = True,
    ) -> Union[torch.Tensor, np.ndarray]:
        inputs_len = len(inputs)
        embs = []
        encoder = self.query_encoder if is_query else self.doc_encoder
        for i in tqdm(range(0, inputs_len, batch_size)):
            batch_inputs = inputs[i : min(i + batch_size, inputs_len)]

            batch_texts, batch_titles = [], []
            for input_ in batch_inputs:
                batch_texts.append(input_["text"])
                if "title" in input_:
                    batch_titles.append(input_["title"])
            kwargs = {
                "titles": None if len(batch_titles) == 0 else batch_titles,
                "fp16": False,
                "max_length": self.max_length,
                "add_sep": False,
            }
            if is_query:
                tmp_embs = []
                for query in batch_texts:
                    kwargs["query"] = query
                    embedding = encoder.encode(**kwargs)
                    tmp_embs.append(embedding)
                embs.append(np.stack(tmp_embs))
            else:
                kwargs["texts"] = batch_texts
                embedding = encoder.encode(**kwargs)
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
        **kwargs: Dict[str, Any]
    ) -> Union[torch.Tensor, np.ndarray]:
        dict_queries = [{"text": query} for query in queries]
        query_embs = self.encode(
            dict_queries,
            is_query=True,
            batch_size=batch_size,
            convert_to_tensor=convert_to_tensor,
        )
        return query_embs

    def encode_corpus(
        self,
        corpus: List[Dict[str, str]],
        batch_size: int,
        convert_to_tensor: bool = True,
        **kwargs: Dict[str, Any]
    ) -> Union[torch.Tensor, np.ndarray]:
        corpus_emb = self.encode(
            corpus,
            is_query=False,
            batch_size=batch_size,
            convert_to_tensor=convert_to_tensor,
        )
        return corpus_emb
