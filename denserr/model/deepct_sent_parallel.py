from __future__ import annotations

import logging
import re
import json
import more_itertools
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from gokart.config_params import inherits_config_params
from gokart.target import TargetOnKart

import pandas as pd
import pyterrier as pt
from tqdm import tqdm
import luigi

from denserr.config import DenseErrConfig
from denserr.utils.template import GokartTask
from denserr.model._base import Retriever
from denserr.dataset.load_dataset import LoadDataset
from denserr.utils.util import (
    aggregate_sentences,
    breakup_to_sentenses,
    index_base_path,
    project_dir,
    simple_batching,
    marge_files,
)


logger = logging.getLogger(__name__)

try:
    pt.init(mem=50_000)
    from pyterrier_deepct import DeepCT, Toks2Text
except ModuleNotFoundError:
    logger.warning(
        "DeepCT module does not found. You need to fix this if you are trying to use DeepctSentRetriever."
    )


def deepct_inference_in_parallel(data, device, window_size):
    deepct = DeepCT(device=device)
    deepct_indexer = (
        pt.apply.generic(breakup_to_sentenses("df", window_size=window_size))
        >> deepct
        >> Toks2Text()
    )

    result_df = deepct_indexer.transform(pd.DataFrame(list(data)))
    return result_df


class ParallelDeepctSent(pt.Transformer):
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
                    deepct_inference_in_parallel,
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

        print("concatnating dataframe")
        result_df = pd.concat(dfs)
        print(f"g df (length: {len(result_df)})")
        return result_df


class DeepctSentParallelRetriever(Retriever):
    _instance = None

    def __init__(self) -> None:
        raise NotImplementedError("Cannot Generate Instance By Constructor")

    @classmethod
    def get_instance(cls) -> DeepctSentParallelRetriever:
        if cls._instance is None:
            if not pt.started():
                pt.init(mem=100_000)
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
            # indexer.setProperty("max.term.length", "200000000")
            # indexer.setProperty("indexer.meta.forward.keylens", "200000000")
            return indexer

        self.window_size = max(1, window_size)
        self.index_prefix = index_prefix
        self.dataset_name = dataset_name

        logger.info(
            (
                f"DeepctSentParallelRetriever(dataset_name: {dataset_name}, ",
                f"window_size: {window_size}, index_prefix: {index_prefix})",
            )
        )
        self.model_name = (
            "deepct-sent-parallel"
            if self.window_size == 1
            else f"deepct-sent-parallel-w{self.window_size}"
        )

        self.index_path = project_dir.joinpath(
            "index", index_prefix, self.model_name, dataset_name
        )
        self.tmp_index_path = project_dir.joinpath(
            "index", index_prefix, self.model_name, "tmp"
        )

        device = "cuda:0"

        batch_size = 16
        max_workers = 1

        self.indexer_pipe = _indexer(self.index_path, threads=1)

        self.deepct = DeepCT(device=device)
        self.deepct_indexer = self.deepct >> Toks2Text()
        self.tokenizer = self.deepct.tokenizer
        self.tmp_indexer_pipe = (
            pt.apply.generic(breakup_to_sentenses("df", window_size=self.window_size))
            >> self.deepct_indexer
            >> _indexer(self.tmp_index_path, 1)
        )
        self.topk = topk

    def load_processed_corpus(self) -> Iterable[Dict[str, str]]:
        output_path_base = project_dir.joinpath(
            "cache",
            "PreprocessDeepCTSentParallel",
            f"{self.dataset_name}/{self.model_name}",
            self.model_name,
        )
        with output_path_base.open() as f:
            for line in tqdm(f, total=32257573):
                line = (
                    line.replace(r"\u0000", "").replace(r"\0", "").replace(r"\x00", "")
                )
                doc = json.loads(line)
                yield doc

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
        indexref = indexer.index(corpus_iter)
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
            # print(x.head())
            print(x.info(verbose=verbose, memory_usage=memory_usage))
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
                self.load_processed_corpus(),
                custom_indexer=custom_indexer,
            )

        self.index = pt.IndexFactory.of(indexref)

        topics = pd.DataFrame.from_dict(
            {"qid": queries.keys(), "query": queries.values()}
        )
        topics = self.preprocess_topics(topics)
        batch_retrieve = pt.BatchRetrieve(
            self.index,
            num_results=self.topk,
            wmodel="BM25",
        )
        retr_pipe = batch_retrieve >> pt.apply.generic(aggregate_sentences)
        logger.info(f"retrieving with {len(queries)} queries ...")
        result = retr_pipe.transform(topics)

        return result

    def get_textscorer(self):
        # print(":::::::::: get_textscorer :::::::::::::")
        # print(f"self.index_path: {self.index_path}")
        # print(":::::::::::::::::::::::::::::::::::::::")
        textscorer = (
            pt.apply.generic(breakup_to_sentenses("df", window_size=self.window_size))
            >> self.deepct
            >> Toks2Text()
            >> pt.text.scorer(
                takes="docs",
                body_attr="text",
                wmodel="BM25",
                background_index=str(self.index_path),
            )
            >> pt.apply.generic(aggregate_sentences)
        )
        return textscorer

    def single_doc_score(self, query: str, text: str, title: str = "") -> float:
        query = self.query_preprocess(query)
        df = pd.DataFrame(
            [
                ["q1", query, "d1", title + " " + text],
            ],
            columns=["qid", "query", "docno", "text"],
        )
        textscorer = self.get_textscorer()
        result = textscorer.transform(df)
        return result["q1"]["d1"]

    def batch_single_doc_score(
        self, queries: List[str], texts: List[str]
    ) -> List[float]:
        queries = [self.query_preprocess(query) for query in queries]
        records = []
        for i, (query, text) in enumerate(zip(queries, texts)):
            records.append(["q1", query, f"d{i}", " " + text])
        df = pd.DataFrame(
            records,
            columns=["qid", "query", "docno", "text"],
        )
        textscorer = self.get_textscorer()
        result = textscorer.transform(df)
        scores: List[float] = []
        for qid, _, docno, _ in records:
            if docno not in result[qid]:
                print(f"\n\n\n\ndocno {docno} does not existed!\n\n\n\n")
                score = 0.0
            else:
                score = result[qid][docno]
            scores.append(score)
        return scores


def deepct_inference_in_parallel_writeout_file(
    corpus, node_id, window_size, start_at, end_at, output_path
):
    def corpus_iter(docs: Iterable[Dict[str, str]]) -> Iterable[Dict[str, str]]:
        for i, doc in enumerate(tqdm(docs, desc=f"corpus_iter({node_id})")):
            if i >= end_at:
                print(f"*** {node_id} Break at {i} ***")
                break

            if i < start_at:
                continue

            yield {"docno": doc["id"], "text": doc["text"]}

    deepct = DeepCT(device=f"cuda:{node_id}")
    deepct_indexer = (
        pt.apply.generic(breakup_to_sentenses("df", window_size=window_size))
        >> deepct
        >> Toks2Text()
    )

    write_count = 0
    with output_path.open("w") as f:
        for batch in more_itertools.batched(corpus_iter(corpus), 100):
            result_df = deepct_indexer.transform(pd.DataFrame(list(batch)))
            for row in result_df.itertuples(index=False):
                f.write(json.dumps(row._asdict(), ensure_ascii=False) + "\n")
                write_count += 1

    return write_count


@inherits_config_params(DenseErrConfig)
class PreprocessDeepCTSentParallel(GokartTask):
    dataset_name = luigi.Parameter()
    model_name = luigi.Parameter()

    use_pyterrier = luigi.BoolParameter()
    window_size = luigi.IntParameter(1)

    def requires(self) -> Tuple[GokartTask, GokartTask]:
        return LoadDataset()

    def output(self) -> TargetOnKart:
        return self.make_target(
            f"denserr/preprocess/PreprocessDeepCTSentParallel/{self.dataset_name}/{self.model_name}.pkl"
        )

    def preprocess(self, corpus):
        output_path_base = project_dir.joinpath(
            "cache",
            "PreprocessDeepCTSentParallel",
            f"{self.dataset_name}/{self.model_name}",
            self.model_name,
        )
        output_path_base.parent.mkdir(parents=True, exist_ok=True)
        logger.info(
            (
                f"DeepctSentParallelRetriever(dataset_name: {self.dataset_name}, ",
                f"window_size: {self.window_size}, index_prefix: {self.output()})",
            )
        )

        max_workers = 4

        corpus_len = len(corpus)
        futures = []
        output_pathes = []
        with ProcessPoolExecutor(
            max_workers=max_workers, mp_context=mp.get_context("spawn")
        ) as executor:
            for i in range(max_workers):
                start_at = i * corpus_len // max_workers
                end_at = min(corpus_len, (i + 1) * corpus_len // max_workers)

                output_path_filename = f"tmp.{output_path_base.stem}.{i}"
                output_path = output_path_base.parent.joinpath(output_path_filename)

                future = executor.submit(
                    deepct_inference_in_parallel_writeout_file,
                    corpus,
                    i,
                    self.window_size,
                    start_at,
                    end_at,
                    output_path,
                )
                logger.info(f"queued {i}")
                output_pathes.append(output_path)
                futures.append(future)

        logger.info("waiting all process completed...")
        for future in as_completed(futures):
            path = future.result()
        logger.info("done all process")

        write_count = marge_files(output_pathes, output_path_base)
        logger.info(f"preprocessed corpus length: {write_count}")

    def corpus_iter(self, docs: Iterable[Dict[str, str]]) -> Iterable[Dict[str, str]]:
        for doc in tqdm(docs, desc="corpus_iter"):
            yield {"docno": doc["id"], "text": doc["text"]}

    def run(self) -> None:
        corpus, _, _ = self.load()
        self.preprocess(corpus)
