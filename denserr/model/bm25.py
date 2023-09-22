from __future__ import annotations

import re
from collections import defaultdict
from logging import getLogger
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
import pyterrier as pt
from tqdm import tqdm

from denserr.model._base import Retriever
from denserr.utils.util import project_dir

logger = getLogger(__name__)


class BM25Retriever(Retriever):
    _instance = None

    def __init__(self) -> None:
        raise NotImplementedError("Cannot Generate Instance By Constructor")

    @classmethod
    def get_instance(cls) -> BM25Retriever:
        if not cls._instance:
            if not pt.started():
                pt.init(mem=60_000)
            cls._instance = cls.__new__(cls)

        return cls._instance

    def set_params(
        self,
        dataset_name: str,
        model_name: str,
        topk: int = 10000,
        index_prefix: str = "",
        config: Optional[Dict] = None,
    ) -> None:
        self.index_path = project_dir.joinpath(
            "index", index_prefix, model_name, dataset_name
        )
        self.topk = topk
        self.control = config
        logger.info(f"control: {self.control}")

    def load_indexref(self) -> Optional[str]:
        if not hasattr(self, "index_path"):
            raise Exception("You need to call set_params in advance to set index path")

        data_property_path = self.index_path.joinpath("data.properties")
        if data_property_path.exists():
            indexref = str(data_property_path)
            return indexref
        else:
            return None

    def indexing(self, corpus_iter: Iterable, fields: List[str] = ["text"]) -> None:
        logger.info("indexing ...")
        indexref = self.load_indexref()

        if indexref is None:
            indexer = pt.index.IterDictIndexer(
                str(self.index_path), threads=16, blocks=True, overwrite=True
            )
            indexref = indexer.index(corpus_iter, fields=fields)
        self.index = pt.IndexFactory.of(indexref)

    def query_preprocess(self, query: str) -> str:
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
        for doc in tqdm(docs, desc="BM25 corpus iter"):
            # print(doc)
            yield {"docno": doc["id"], "text": doc["text"]}

    def set_index(self) -> None:
        if not hasattr(self, "index") or self.index is None:
            indexref = self.load_indexref()
            if indexref is None:
                raise Exception(
                    "Need background index for calculating single doc score"
                )

            self.index = pt.IndexFactory.of(indexref)

    def get_bm25(self) -> pt.Transformer:
        self.set_index()

        if self.control is None:
            return pt.BatchRetrieve(
                self.index,
                num_results=self.topk,
                wmodel="BM25",
                verbose=True,
                threads=2,
            )
        else:
            return pt.BatchRetrieve(
                self.index,
                num_results=self.topk,
                wmodel="BM25",
                controls=self.control,
                verbose=True,
                threads=2,
            )

    def retrieve(
        self,
        corpus: Dict[str, Dict[str, str]],
        queries: Dict[str, str],
        **kwargs: Dict[Any, Any],
    ) -> Dict[str, Dict[str, float]]:
        logger.info("retriving by BM25")
        self.indexing(
            self.corpus_iter(corpus),
        )
        topics = pd.DataFrame.from_dict(
            {"qid": queries.keys(), "query": queries.values()}
        )
        topics = self.preprocess_topics(topics)
        bm25 = self.get_bm25()
        result_df = bm25.transform(topics)

        result: defaultdict = defaultdict(dict)
        for _, row in result_df.iterrows():
            result[row["qid"]][row["docno"]] = row["score"]
        return result

    def single_doc_score(self, query: str, text: str, title: str = "") -> float:
        if not hasattr(self, "index") or self.index is None:
            indexref = self.load_indexref()
            if indexref is None:
                raise Exception(
                    "Need background index for calculating single doc score"
                )

            self.index = pt.IndexFactory.of(indexref)

        query = self.query_preprocess(query)
        df = pd.DataFrame(
            [
                ["q1", query, "d1", title + " " + text],
            ],
            columns=["qid", "query", "docno", "text"],
        )
        textscorer = (
            pt.batchretrieve.TextScorer(
                takes="docs",
                body_attr="text",
                background_index=self.index,
                wmodel="BM25",
            )
            if self.control is None
            else pt.batchretrieve.TextScorer(
                takes="docs",
                body_attr="text",
                background_index=self.index,
                wmodel="BM25",
                controls=self.control,
            )
        )
        rtr = textscorer.transform(df)
        score: float = rtr.at[0, "score"]
        return score
