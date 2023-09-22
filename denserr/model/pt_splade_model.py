from __future__ import annotations

from typing import Iterable, List, Any, Optional, Dict, Tuple
from collections import defaultdict
from tqdm import tqdm
import re
import logging
from pathlib import Path

import pandas as pd
import pyterrier as pt
import pyt_splade
from concurrent.futures import ProcessPoolExecutor, as_completed

from denserr.utils.util import project_dir
from denserr.model._base import Retriever

logger = logging.getLogger(__name__)


class PtSpladeRetriever(Retriever):
    _instance = None

    def __init__(self) -> None:
        raise NotImplementedError("Cannot Generate Instance By Constructor")

    @classmethod
    def get_instance(cls) -> PtSpladeRetriever:
        if not cls._instance:
            if not pt.started():
                pt.init()
            cls._instance = cls.__new__(cls)

        return cls._instance

    def set_params(
        self, dataset_name: str, topk: int = 10000, index_prefix: str = ""
    ) -> None:
        def _indexer(index_path: Path) -> pt.Transformer:
            indexer = pt.IterDictIndexer(str(index_path), overwrite=True)
            indexer.setProperty("termpipelines", "")
            indexer.setProperty("tokeniser", "WhitespaceTokeniser")
            return indexer

        self.index_path = project_dir.joinpath(
            "index", index_prefix, "pt_splade", dataset_name
        )
        self.tmp_index_path = project_dir.joinpath(
            "index", index_prefix, "pt_splade", "tmp"
        )

        self.splade = pyt_splade.SpladeFactory()
        self.splade_indexer = self.splade.indexing() >> pyt_splade.toks2doc()
        self.tokenizer = self.splade.tokenizer

        self.indexer_pipe = self.splade_indexer >> _indexer(self.index_path)
        self.tmp_indexer_pipe = self.splade_indexer >> _indexer(self.tmp_index_path)
        self.topk = topk

    def load_indexref(self, index_path: Path) -> Optional[str]:
        if not hasattr(self, "index_path"):
            raise Exception("You need to call set_params in advance to set index path")

        data_property_path = index_path.joinpath("data.properties")
        if data_property_path.exists():
            indexref = str(data_property_path)
            return indexref
        else:
            return None

    def indexing(
        self,
        corpus_iter: Iterable,
        custom_indexer: pt.Transformer = None,
        fields: List[str] = ["text"],
    ) -> pt.IndexRef:
        logger.info("indexing ...")
        indexer = self.indexer_pipe if custom_indexer is None else custom_indexer
        indexref = indexer.index(corpus_iter, batch_size=32)
        return indexref

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

    def debug_pipe(self, verbose=False, memory_usage=True):
        def _inner(x: pd.DataFrame) -> pd.DataFrame:
            print(x.head())
            # print(x.info(verbose=verbose, memory_usage=memory_usage))
            return x

        return _inner

    def corpus_iter(self, docs: Iterable[Dict[str, str]]) -> Iterable[Dict[str, str]]:
        for doc in docs:
            yield {"docno": doc["id"], "text": doc["text"]}

    def get_textscorer(self) -> pt.batchretrieve.TextScorer:
        if not hasattr(self, "_textscorer"):
            textscorer = (
                self.splade.query()
                >> self.splade_indexer
                >> pt.text.scorer(
                    takes="docs",
                    body_attr="text",
                    wmodel="Tf",
                )
            )
            self._textscorer = textscorer
        return self._textscorer

    def retrieve(
        self,
        corpus: Iterable[Dict[str, str]],
        queries: Dict[str, str],
        overwrite_index: bool = False,
        custom_indexer: pt.Transformer = None,
        **kwargs: Dict[Any, Any],
    ) -> Dict[str, Dict[str, float]]:
        indexref = self.load_indexref(self.index_path)
        if overwrite_index or indexref is None:
            indexref = self.indexing(
                self.corpus_iter(corpus),
                custom_indexer=custom_indexer,
            )
        self.index = pt.IndexFactory.of(indexref)

        topics = pd.DataFrame.from_dict(
            {"qid": queries.keys(), "query": queries.values()}
        )
        topics = self.preprocess_topics(topics)
        batch_retrieve = pt.BatchRetrieve(
            self.index, num_results=self.topk, wmodel="Tf"
        )
        query_splade = self.splade.query()
        retr_pipe = query_splade >> batch_retrieve
        result_df = retr_pipe.transform(topics)

        result: defaultdict = defaultdict(dict)
        for _, row in result_df.iterrows():
            result[row["qid"]][row["docno"]] = row["score"]
        return result

    def single_doc_score(self, query: str, text: str, title: str = "") -> float:
        corpus = [{"text": text, "id": "d1"}]
        queries = {"q1": query}
        result = self.retrieve(
            corpus, queries, overwrite_index=True, custom_indexer=self.tmp_indexer_pipe
        )
        return result["q1"]["d1"]

    def batch_single_doc_score(
        self, queries: List[str], texts: List[str]
    ) -> List[float]:
        docid_to_text = {}
        corpus: List[Dict[str, str]] = []
        queries_dict: Dict[str, str] = {}
        query_doc_pairs: List[Tuple[str, str]] = []
        for i, (query, text) in enumerate(zip(queries, texts)):
            qid = f"q_{i}"
            docid = f"d_{i}"
            queries_dict[qid] = query
            corpus.append({"id": docid, "text": text})
            docid_to_text[docid] = text
            query_doc_pairs.append((qid, docid))
        result = self.retrieve(
            corpus,
            queries_dict,
            overwrite_index=True,
            custom_indexer=self.tmp_indexer_pipe,
        )
        scores: List[float] = []
        for qid, docid in query_doc_pairs:
            if docid not in result[qid]:
                score = 0.0
            else:
                score = result[qid][docid]
            scores.append(score)

        return scores
