from __future__ import annotations

import logging
import re
import more_itertools
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

import pandas as pd
import pyterrier as pt

if not pt.started():
    pt.init()
import pyt_splade
from tqdm import tqdm

from denserr.model._base import Retriever
from denserr.utils.util import (
    aggregate_sentences,
    breakup_to_sentenses,
    index_base_path,
    project_dir,
    simple_batching,
)

logger = logging.getLogger(__name__)


def splade_inference_in_parallel(data, device, window_size):
    splade = pyt_splade.SpladeFactory(device=device)
    splade_indexing = pt.apply.generic(simple_batching(splade.indexing(), 16))
    splade_indexer = (
        pt.apply.generic(breakup_to_sentenses("list", window_size=window_size))
        >> splade_indexing
        >> pyt_splade.toks2doc()
    )

    result_df = splade_indexer.transform(pd.DataFrame(list(data)))
    return result_df


class ParallelSplade(pt.Transformer):
    def __init__(
        self, batch_size: int = 16, max_workers: int = 4, window_size: int = 1
    ):
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.window_size = window_size
        logger.info(
            f"max_workers: {self.max_workers}, batch_size: {self.batch_size}, window_size: {self.window_size}"
        )

    def transform_iter(self, inp: Iterable[dict]) -> pd.DataFrame:
        logger.info(f"transform_iter: inp len {len(inp)}\n")
        futures = []
        with ProcessPoolExecutor(
            max_workers=self.max_workers, mp_context=mp.get_context("spawn")
        ) as executor:
            for i, inp_chunk in enumerate(more_itertools.divide(self.max_workers, inp)):
                logger.info(f"queing task at {i}")
                device = f"cuda:{i}"

                future = executor.submit(
                    splade_inference_in_parallel,
                    inp_chunk,
                    device,
                    self.window_size,
                )
                futures.append(future)

        dfs = []
        for future in tqdm(
            as_completed(futures), desc="waiting futures", total=len(futures)
        ):
            df = future.result()
            dfs.append(df)

        logger.info("concatnating dataframe")
        result_df = pd.concat(dfs)
        logger.info(f"return df (length: {len(result_df)})")
        return result_df


class PtSpladeSentParallelRetriever(Retriever):
    _instance = None

    def __init__(self) -> None:
        raise NotImplementedError("Cannot Generate Instance By Constructor")

    @classmethod
    def get_instance(cls) -> PtSpladeSentParallelRetriever:
        if not cls._instance:
            if not pt.started():
                pt.init()
            cls._instance = cls.__new__(cls)

        return cls._instance

    def set_params(
        self,
        dataset_name: str,
        topk: int = 10000,
        window_size: int = 1,
        index_prefix: str = "",
    ) -> None:
        def _indexer(index_path: Path, threads: int = 32) -> pt.Transformer:
            indexer = pt.IterDictIndexer(
                str(index_path), verbose=True, overwrite=True, threads=threads
            )
            indexer.setProperty("termpipelines", "")
            indexer.setProperty("tokeniser", "WhitespaceTokeniser")
            return indexer

        self.window_size = max(1, window_size)
        self.index_prefix = index_prefix
        self.dataset_name = dataset_name

        logger.info(
            (
                f"PtSpladeSentParallelRetriever(dataset_name: {dataset_name}, ",
                f"window_size: {window_size}, index_prefix: {index_prefix})",
            )
        )
        model_name = (
            "pt_splade_parallel_sent"
            if self.window_size == 1
            else f"pt_splade_parallel_sent_w{self.window_size}"
        )

        self.index_path = project_dir.joinpath(
            "index", index_prefix, model_name, dataset_name
        )
        self.tmp_index_path = project_dir.joinpath(
            "index", index_prefix, model_name, "tmp"
        )

        device = "cuda:0"

        batch_size = 16
        max_workers = 4

        self.indexer_pipe = ParallelSplade(
            batch_size=batch_size, max_workers=max_workers, window_size=window_size
        ) >> _indexer(self.index_path)

        self.splade = pyt_splade.SpladeFactory(device=device)
        splade_indexing = pt.apply.generic(simple_batching(self.splade.indexing(), 16))
        self.splade_indexer = splade_indexing >> pyt_splade.toks2doc()
        self.tokenizer = self.splade.tokenizer
        self.tmp_indexer_pipe = (
            pt.apply.generic(breakup_to_sentenses("list", window_size=self.window_size))
            >> self.splade_indexer
            >> _indexer(self.tmp_index_path, 1)
        )
        self.topk = topk

    def load_indexref(self, index_path: Path) -> Optional[str]:
        data_property_path = index_path.joinpath("data.properties")
        if data_property_path.exists():
            logger.info(f"existing index found. load from {data_property_path}")
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
        logger.info(
            f"indexing {'' if custom_indexer is None else 'with custom_indexer'} ..."
        )
        indexer = self.indexer_pipe if custom_indexer is None else custom_indexer
        indexref = indexer.index(corpus_iter, batch_size=50_000)
        logger.info("done indexing")
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
        for doc in tqdm(docs, desc="corpus_iter"):
            yield {"docno": doc["id"], "text": doc["text"]}

    def retrieve(
        self,
        corpus: Iterable[Dict[str, str]],
        queries: Dict[str, str],
        overwrite_index: bool = False,
        custom_indexer: pt.Transformer = None,
        **kwargs: Dict[Any, Any],
    ) -> Dict[str, Dict[str, float]]:
        indexref = self.load_indexref(self.index_path)
        if overwrite_index or (indexref is None) or (custom_indexer is not None):
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
        retr_pipe = (
            query_splade >> batch_retrieve >> pt.apply.generic(aggregate_sentences)
        )
        logger.info(f"retrieving with {len(queries)} queries ...")
        result = retr_pipe.transform(topics)

        # result: defaultdict = defaultdict(dict)
        # for _, row in result_df.iterrows():
        #     result[row["qid"]][row["docno"]] = row["score"]
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
        self.set_params(
            self.dataset_name, self.topk, self.window_size, self.index_prefix
        )
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
