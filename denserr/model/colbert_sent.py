from collections import defaultdict
from logging import getLogger
from pathlib import Path
from typing import Any, Dict, Iterable, List

import nltk
import pyterrier as pt
import pandas as pd
from pyterrier.transformer import TransformerBase
from tqdm import tqdm

from denserr.model._base import Retriever
from denserr.utils.util import (
    aggregate_sentences,
    breakup_to_sentenses,
    index_base_path,
)

logger = getLogger(__name__)


class ColbertSentRetriever(Retriever):
    def __init__(
        self,
        model_path: str,
        dataset_name: str,
        model_name: str,
        index_prefix: str = "",
        topk: int = 10000,
        chunksize: int = 64,
        window_size: int = 1,
    ):
        if not pt.started():
            pt.init()
        from pyterrier_colbert.indexing import ColBERTIndexer
        from pyterrier_colbert.ranking import ColBERTFactory

        self.model_index_path = index_base_path.joinpath(index_prefix, model_name)
        self.model_path = model_path
        self.chunksize = chunksize
        self.indexer = ColBERTIndexer(
            model_path, self.model_index_path, dataset_name, chunksize=chunksize
        )
        self.index_path = Path(self.indexer.args.index_path)
        self.pytcolbert = ColBERTFactory(
            model_path,
            self.model_index_path,
            dataset_name,
            faiss_partitions=self.indexer.args.partitions,
            gpu=True,
        )
        self.tokenizer = self.pytcolbert.args.inference.doc_tokenizer.tok
        self.window_size = window_size
        logger.info(f"ColbertSentRetriever(window_size={self.window_size})")

    def debug_pipe(self, x: pd.DataFrame) -> pd.DataFrame:
        print(x.info())
        print(x.head())
        return x

    def indexing_pipeline(self) -> pt.Transformer:
        return (
            pt.apply.generic(breakup_to_sentenses("df", window_size=self.window_size))
            >> self.indexer
        )

    def textscorer(self) -> pt.Transformer:
        return (
            pt.apply.generic(breakup_to_sentenses("df", window_size=self.window_size))
            >> self.pytcolbert.text_scorer()
            >> pt.apply.generic(aggregate_sentences)
        )

    def indexing(
        self,
        corpus_iter: Iterable,
        overwrite: bool = False,
        fields: List[str] = ["text", "title"],
    ) -> None:
        faiss_index_path = self.index_path.joinpath("ivfpq.100.faiss")
        if faiss_index_path.exists():
            logger.info(f"faiss index found. Use existsing index : {faiss_index_path}")
            return None
        self.indexing_pipeline().index(corpus_iter)

    def query_preprocess(self, query: str) -> str:
        query = query.replace("/", " ")
        query = query.replace("'", " ")
        query = query.replace("\n", " ")
        query = query.replace("?", " ")
        query = query.replace(")", "")
        query = query.replace("(", "")
        query = query.replace(":", "")
        return query

    def corpus_iter(self, docs: Iterable[dict]) -> Iterable[Dict[str, str]]:
        for doc in tqdm(docs):
            if not doc["text"]:
                continue
            yield {"docno": doc["id"], "text": doc["text"]}

    def preprocess_topics(self, topics: pd.DataFrame) -> pd.DataFrame:
        topics = topics.rename(columns={"title": "query"})
        topics["query"] = topics["query"].map(self.query_preprocess)
        return topics

    def retrieve(
        self,
        corpus: Iterable,
        queries: Dict[str, str],
        **kwargs: Dict[Any, Any],
    ) -> Dict[str, Dict[str, float]]:
        return self.__retrieve(corpus, queries, overwrite=False)

    def __retrieve(
        self,
        corpus: Iterable[dict],
        queries: Dict[str, str],
        overwrite: bool = False,
    ) -> Dict[str, Dict[str, float]]:
        self.indexing(self.corpus_iter(corpus), overwrite=True)
        retriever = self.pytcolbert.end_to_end()

        topics = pd.DataFrame.from_dict(
            {"qid": queries.keys(), "query": queries.values()}
        )
        topics = self.preprocess_topics(topics)
        result_df = retriever.transform(topics)
        print("aggregating sentences results")
        result = aggregate_sentences(result_df)

        return result

    def single_doc_score(self, query: str, text: str) -> float:
        text_scorer = self.textscorer()
        query = self.query_preprocess(query)
        df = pd.DataFrame(
            [
                ["q1", query, "d1", text],
            ],
            columns=["qid", "query", "docno", "text"],
        )
        result = text_scorer.transform(df)

        score: float = result["q1"]["d1"]
        return score
