from __future__ import annotations

import math
import multiprocessing as mp
import re
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from logging import getLogger
from pathlib import Path
from types import FunctionType
from typing import Any, Dict, Iterable, List, Optional

import nltk
import pandas as pd
import pyterrier as pt
from tqdm import tqdm

from denserr.model._base import Retriever
from denserr.utils.util import (
    aggregate_sentences,
    breakup_to_sentenses,
    index_base_path,
    project_dir,
)

logger = getLogger(__name__)

try:
    pt.init(mem=300_000)
    from pyterrier_deepct import DeepCT, Toks2Text
except ModuleNotFoundError:
    logger.warning(
        "DeepCT module does not found. You need to fix this if you are trying to use DeepctSentRetriever."
    )


def multi_deepset_sent(inp: pd.DataFrame, device_num: int) -> pd.DataFrame:
    deepct = DeepCT(device=f"cuda:{device_num}")
    transformed = deepct.transform(inp)
    return transformed


class ParallelDeepctTransformer(pt.Transformer):
    def __init__(self, devices: List[int] = [0, 1, 2, 3]) -> None:
        self.devices = devices

    def transform(self, inp: pd.DataFrame) -> pd.DataFrame:
        num_worker = len(self.devices)

        data_len = inp.shape[0]

        dfs = []
        futures = []
        with ProcessPoolExecutor(
            max_workers=num_worker, mp_context=mp.get_context("spawn")
        ) as executor:
            for i in range(num_worker):
                start_at = 0 if i == 0 else math.ceil(i * data_len / num_worker)
                end_at = min(data_len, math.ceil((i + 1) * data_len / num_worker))
                inp_partition = inp.loc[start_at:end_at, :]
                device = self.devices[i]

                future = executor.submit(multi_deepset_sent, inp_partition, device)
                futures.append(future)

            logger.info("waiting all process completed...")
            for future in as_completed(futures):
                df = future.result()
                dfs.append(df)
            logger.info("done all process")

        return pd.concat(dfs)


class DeepctSentRetriever(Retriever):
    _instance = None

    def __init__(self) -> None:
        raise NotImplementedError("Cannot Generate Instance By Constructor")

    @classmethod
    def get_instance(cls) -> DeepctSentRetriever:
        if cls._instance is None:
            if not pt.started():
                pt.init(mem=20_000)
            cls._instance = cls.__new__(cls)
        return cls._instance

    def set_params(
        self, dataset_name: str, topk: int = 10000, index_dir: str = ""
    ) -> None:
        self.index_path = project_dir.joinpath(
            index_dir, "index", "deepct-sent", dataset_name
        )
        self.topk = topk
        self.deepct = DeepCT(device="cuda:0")
        # self.parallel_deepct = ParallelDeepctTransformer()
        self.tokenizer = self.deepct.tokenizer

    def query_tokenize(self, text: str) -> str:
        toks = self.tokenizer.tokenize(text)
        fulltokens: List[str] = []
        for token in toks:
            if token.startswith("##"):
                fulltokens[-1] += token[2:]
            else:
                fulltokens.append(token)
        return " ".join(fulltokens)

    def get_textscorer(self) -> pt.batchretrieve.TextScorer:
        if not hasattr(self, "_textscorer"):
            textscorer = (
                pt.apply.generic(self.preprocess_topics)
                >> self.deepct_sent_pipeline()
                >> pt.text.scorer(
                    takes="docs",
                    body_attr="text",
                    wmodel="BM25",
                    background_index=str(self.index_path),
                    properties={"termpipelines": "Stopwords,PorterStemmer"},
                )
                >> pt.apply.generic(aggregate_sentences)
            )
            self._textscorer = textscorer
        return self._textscorer

    def load_indexref(self) -> Optional[str]:
        if not hasattr(self, "index_path"):
            raise Exception("You need to call set_params in advance to set index path")

        data_property_path = self.index_path.joinpath("data.properties")
        if data_property_path.exists():
            indexref = str(data_property_path)
            return indexref
        else:
            return None

    def deepct_sent_pipeline(self) -> pt.Transformer:
        return (
            pt.apply.generic(breakup_to_sentenses("df")) >> self.deepct >> Toks2Text()
        )

    def indexing(
        self,
        corpus_iter: Iterable,
        fields: List[str] = ["text"],
        custum_index_path: Optional[Path] = None,
    ) -> None:
        index_path = self.index_path if custum_index_path is None else custum_index_path
        indexer = self.deepct_sent_pipeline() >> pt.index.IterDictIndexer(
            str(index_path), threads=16, blocks=True, overwrite=True
        )
        indexref = indexer.index(corpus_iter, batch_size=50_000)

    def query_preprocess(self, query: str) -> str:
        query = self.query_tokenize(query)
        code = re.compile(
            "[!\"#$%&'\\\\()*+,-./:;<=>?@[\\]^_`{|}~「」〔〕“”〈〉『』【】＆＊・（）＄＃＠。、？！｀＋￥％]"
        )
        query = code.sub("", query)
        query = query.replace("\n", " ")
        return query

    def preprocess_topics(self, topics: pd.DataFrame) -> pd.DataFrame:
        topics = topics.rename(columns={"title": "query"})
        topics["query"] = topics["query"].map(self.query_preprocess)
        return topics

    def corpus_iter(self, docs: Dict[str, Dict[str, str]]) -> Iterable[Dict[str, str]]:
        for doc in tqdm(docs):
            yield {"docno": doc["id"], "text": doc["text"]}

    def debug_pipe(self, x: pd.DataFrame) -> pd.DataFrame:
        print(x.info())
        return x

    def retrieve(
        self,
        corpus: Dict[str, Dict[str, str]],
        queries: Dict[str, str],
        custum_index_path: Optional[str] = None,
        **kwargs: Dict[Any, Any],
    ) -> Dict[str, Dict[str, float]]:
        index_path = self.index_path if custum_index_path is None else custum_index_path
        self.indexing(self.corpus_iter(corpus), custum_index_path=index_path)
        topics = pd.DataFrame.from_dict(
            {"qid": queries.keys(), "query": queries.values()}
        )
        bm25 = (
            pt.apply.generic(self.preprocess_topics)
            >> pt.BatchRetrieve(
                str(index_path),
                num_results=self.topk,
                wmodel="BM25",
                properties={"termpipelines": "Stopwords,PorterStemmer"},
            )
            >> pt.apply.generic(aggregate_sentences)
        )
        result = bm25.transform(topics)

        return result

    def single_doc_score(self, query: str, text: str, title: str = "") -> float:
        query = self.query_preprocess(query)
        df = pd.DataFrame(
            [
                ["q1", query, "d1", title + " " + text],
            ],
            columns=["qid", "query", "docno", "text"],
        )
        textscorer = self.get_textscorer()
        rtr = textscorer.transform(df)
        score: float = rtr["q1"]["d1"]
        return score

    def batch_single_doc_score(
        self, queries: List[str], texts: List[str]
    ) -> List[float]:
        # qid is fixed to "q1" since expected as all queries are the same
        queries = [self.query_preprocess(query) for query in queries]
        df = pd.DataFrame(
            [
                ["q1", query, f"d{i}", " " + text]
                for i, (query, text) in enumerate(zip(queries, texts))
            ],
            columns=["qid", "query", "docno", "text"],
        )
        textscorer = self.get_textscorer()
        result = textscorer.transform(df)
        sorted_result = sorted(result["q1"].items(), key=lambda x: x[0])
        scores: List[float] = [score for docid, score in sorted_result]
        return scores
