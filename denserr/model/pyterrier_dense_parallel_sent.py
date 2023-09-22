from __future__ import annotations

import multiprocessing as mp
import os
import pickle
import re
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from logging import getLogger
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, List, Optional, Union

import nltk
from tqdm import tqdm
import torch
import faiss
import more_itertools
import numpy as np
import pandas as pd
import pyterrier as pt
from pyterrier.model import add_ranks
from pyterrier.transformer import TransformerBase

from denserr.model._base import Retriever
from denserr.model.pyterrier_dense_parallel import (
    PyTParallelDenseIndexer,
    PyTParallelDenseRetrieval,
)
from denserr.utils.util import (
    index_base_path,
    breakup_to_sentenses,
    aggregate_sentences,
)

logger = getLogger(__name__)


class PyTParallelDenseSentRetriever(Retriever):
    _instance = None

    def __init__(
        self,
        load_ance: Any,
        dataset_name: str,
        model_name: str,
        index_prefix: str = "",
        topk: int = 10000,
        segment_size: int = 500_000,
        window_size: int = 1,
    ) -> None:
        if not pt.started():
            pt.init()

        self.index_path = index_base_path.joinpath(
            index_prefix, model_name, dataset_name
        )
        self.topk = topk
        self.window_size = window_size

        self.indexer = PyTParallelDenseIndexer(
            load_ance, self.index_path, num_docs=topk, segment_size=segment_size
        )
        self.index_pipe = (
            pt.apply.generic(breakup_to_sentenses("df", window_size=self.window_size))
            >> self.indexer
        )
        self.encoder = load_ance("cuda:0")
        self.retriever = PyTParallelDenseRetrieval(
            self.encoder, self.index_path, dataset_name, model_name
        )
        self.tokenizer = self.encoder.tokenizer
        logger.info(
            (
                f"PyTParallelDenseSentRetriever(dataset_name: {dataset_name}, ",
                f"window_size: {self.window_size}, index_prefix: {index_prefix})",
            )
        )

    def indexing(
        self,
        corpus_iter: Iterable,
        indexer: TransformerBase,
        overwrite: bool = False,
        fields: List[str] = ["text", "title"],
    ) -> None:
        index_path = self.index_path
        if not overwrite and index_path.joinpath("shards.pkl").exists():
            logger.info(f"shards.pkl found. Use existsing index : {index_path}")
            return None
        logger.info(f"indexing with index_path: {index_path}")
        self.index_pipe.index(corpus_iter)

    def query_preprocess(self, query: str) -> str:
        query = query.replace("/", " ")
        query = query.replace("'", " ")
        query = query.replace("\n", " ")
        query = query.replace("?", " ")
        query = query.replace(")", "")
        query = query.replace("(", "")
        query = query.replace(":", "")
        return query

    def preprocess_topics(self, topics: pd.DataFrame) -> pd.DataFrame:
        topics = topics.rename(columns={"title": "query"})
        topics["query"] = topics["query"].map(self.query_preprocess)
        return topics

    def corpus_iter(self, docs: Iterable[dict]) -> Iterable[Dict[str, str]]:
        for doc in docs:
            yield {"docno": doc["id"], "text": doc["text"]}

    def retrieve(
        self,
        corpus: Iterable,
        queries: Dict[str, str],
        **kwargs: Dict[Any, Any],
    ) -> Dict[str, Dict[str, float]]:
        return self.__retrieve(corpus, queries, self.index_pipe, overwrite=False)

    def __retrieve(
        self,
        corpus: Iterable[dict],
        queries: Dict[str, str],
        indexer: TransformerBase,
        overwrite: bool = False,
    ) -> Dict[str, Dict[str, float]]:
        self.indexing(self.corpus_iter(corpus), indexer, overwrite=overwrite)

        topics = pd.DataFrame.from_dict(
            {"qid": queries.keys(), "query": queries.values()}
        )
        topics = self.preprocess_topics(topics)
        result_df = self.retriever.transform(topics)
        print(f"result_df len: {len(result_df)}")
        result = aggregate_sentences(result_df)
        print(f"result len: {len(result)}")
        print(f"result qid[0] ranking len: {len(result[list(result.keys())[0]])}")
        return result

    def single_doc_score(self, query: str, text: str) -> float:
        query = self.query_preprocess(query)
        q_emb = self.encoder.encode_queries([query], 16, convert_to_tensor=False)

        sentences = breakup_to_sentenses("list", window_size=self.window_size)(
            pd.DataFrame.from_records([{"text": text, "docno": "d1"}])
        )
        d_embs = self.encoder.encode_corpus(sentences, 16, convert_to_tensor=False)
        score = max(np.matmul(q_emb, d_embs.T)[0].tolist())
        return float(score)
