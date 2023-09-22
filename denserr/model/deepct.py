from __future__ import annotations

from typing import Iterable, List, Any, Optional, Dict
from collections import defaultdict
import re
from logging import getLogger

import pandas as pd
import pyterrier as pt

from denserr.utils.util import project_dir
from denserr.model._base import Retriever

logger = getLogger(__name__)

try:
    pt.init(mem=300_000)
    from pyterrier_deepct import DeepCT, Toks2Text
except ModuleNotFoundError:
    logger.warning(
        "DeepCT module does not found. You need to fix this if you are trying to use DeepctRetriever."
    )


class DeepctRetriever(Retriever):
    _instance = None

    def __init__(self) -> None:
        raise NotImplementedError("Cannot Generate Instance By Constructor")

    @classmethod
    def get_instance(cls) -> DeepctRetriever:
        print("++++++++++++ not pt.started() ++++++++++++")
        if cls._instance is None:
            print("cls._instance is None")
            if not pt.started():
                print("pt.init")
                pt.init(mem=30_000)
            cls._instance = cls.__new__(cls)
        print("+++++++++++++++++++++++++++++++++++++++++")
        return cls._instance

    def set_params(
        self, dataset_name: str, topk: int = 10000, index_prefix: str = ""
    ) -> None:
        self.index_path = project_dir.joinpath(
            "index", index_prefix, "deepct", dataset_name
        )
        self.topk = topk
        self.deepct = DeepCT()
        self.tokenizer = self.deepct.tokenizer

    def get_textscorer(self) -> pt.batchretrieve.TextScorer:
        # if not hasattr(self, "_textscorer"):
        print(":::::::::: get_textscorer :::::::::::::")
        print(f"self.index_path: {self.index_path}")
        print(":::::::::::::::::::::::::::::::::::::::")
        textscorer = (
            self.deepct
            >> Toks2Text()
            >> pt.text.scorer(
                takes="docs",
                body_attr="text",
                wmodel="BM25",
                background_index=str(self.index_path),
            )
        )
        # self._textscorer = textscorer
        # return self._textscorer
        return textscorer

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
        indexer = (
            self.deepct
            >> Toks2Text()
            >> pt.index.IterDictIndexer(
                str(self.index_path), threads=16, blocks=True, overwrite=True
            )
        )
        indexref = indexer.index(corpus_iter)

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
        for doc in docs:
            yield {"docno": doc["id"], "text": doc["text"]}

    def retrieve(
        self,
        corpus: Dict[str, Dict[str, str]],
        queries: Dict[str, str],
        **kwargs: Dict[Any, Any],
    ) -> Dict[str, Dict[str, float]]:
        self.indexing(
            self.corpus_iter(corpus),
        )
        topics = pd.DataFrame.from_dict(
            {"qid": queries.keys(), "query": queries.values()}
        )
        topics = self.preprocess_topics(topics)
        bm25 = pt.BatchRetrieve(
            str(self.index_path), num_results=self.topk, wmodel="BM25"
        )
        result_df = bm25.transform(topics)

        result: defaultdict = defaultdict(dict)
        for _, row in result_df.iterrows():
            result[row["qid"]][row["docno"]] = row["score"]
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
        score: float = rtr.at[0, "score"]
        return score

    def batch_single_doc_score(
        self, queries: List[str], texts: List[str]
    ) -> List[float]:
        queries = [self.query_preprocess(query) for query in queries]
        df = pd.DataFrame(
            [
                ["q1", query, f"d{i}", " " + text]
                for i, (query, text) in enumerate(zip(queries, texts))
            ],
            columns=["qid", "query", "docno", "text"],
        )
        textscorer = self.get_textscorer()
        rtr = textscorer.transform(df)
        scores: List[float] = rtr["score"]
        return scores
