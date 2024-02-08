import os
from logging import getLogger
from pathlib import Path
import json
import csv
import math
from typing import Generator, Optional, Iterable, Any, Dict, Set
from concurrent.futures import ProcessPoolExecutor, as_completed
import pickle

import pandas as pd
import luigi
import nltk
import polars as pl
from beir.datasets.data_loader import GenericDataLoader
from gokart.config_params import inherits_config_params
from gokart.target import TargetOnKart
from tqdm import tqdm

from denserr.config import DenseErrConfig
from denserr.dataset.msmarco_pas import LoadMSMarcoPassage
from denserr.dataset.msmarco_doc import LoadMSMarcoDocument
from denserr.dataset.robust04 import LoadRobust04
from denserr.dataset.nfcorpus import LoadNFCorpus
from denserr.dataset.scifact import LoadScifact
from denserr.dataset.trec_dl19 import LoadTrecDL19Doc
from denserr.dataset.trec_dl20 import LoadTrecDL20Doc
from denserr.utils.template import GokartTask
from denserr.model.load_model import LoadRetriever
from denserr.utils import util
from ._base import ILoadModel, PolarsCorpusLoader

logger = getLogger(__name__)


AVAILABLE_DATASET: Dict[str, ILoadModel] = {
    "msmarco-pas": LoadMSMarcoPassage(),
    "msmarco-doc": LoadMSMarcoDocument(),
    "dl19-doc": LoadTrecDL19Doc(),
    "dl20-doc": LoadTrecDL20Doc(),
    "robust04": LoadRobust04(),
    "nfcorpus": LoadNFCorpus(),
    "scifact": LoadScifact(),
}


def truncate_by_sentences(
    corpus_loader: ILoadModel,
    start_at: int,
    end_at: int,
    max_doc_len: int,
    tokenizer: Any,
    output_path: Path,
    include_header: bool,
) -> Path:
    corpus = corpus_loader.load_corpus()
    write_count = 0
    with output_path.open("w", newline="") as f:
        fieldnames = ["id", "text"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if include_header:
            writer.writeheader()
        for i, doc in enumerate(corpus):
            if i >= end_at:
                break

            if i < start_at:
                continue

            doc_id = doc["id"]
            if len(doc["text"]) == 0:
                continue

            sentences = nltk.sent_tokenize(doc["text"])
            if len(sentences) <= 0:
                continue

            tokenized = tokenizer(sentences, add_special_tokens=False)

            accepted_ids = []
            accepted_ids_sum = 0
            for ids in tokenized["input_ids"]:
                if accepted_ids_sum + len(ids) < max_doc_len:
                    accepted_ids.append(ids)
                    accepted_ids_sum += len(ids)
                else:
                    break

            restored_sentences = tokenizer.batch_decode(
                accepted_ids, skip_special_tokens=True
            )
            truncated_text = " ".join(restored_sentences)

            new_doc = {
                "id": doc_id,
                "text": json.dumps(truncated_text, ensure_ascii=True),
            }
            writer.writerow(new_doc)
            write_count += 1

    return output_path


@inherits_config_params(DenseErrConfig)
class LoadDataset(GokartTask):
    dataset_name = luigi.Parameter()
    model_name = luigi.Parameter()
    max_doc_len = luigi.IntParameter()

    separator = luigi.Parameter(",")

    def cache_path_base(self, model_name: Optional[str] = None) -> Path:
        model_name = self.model_name if model_name is None else model_name
        return Path(
            f"denserr/dataset/load_dataset/{self.dataset_name}/{model_name}_{self.max_doc_len}"
        )

    def output(self) -> TargetOnKart:
        return self.make_target(f"{self.cache_path_base()}.pkl")

    def cache_csv_path(self, model_name: Optional[str] = None) -> Path:
        path = Path(f"resources/{self.cache_path_base(model_name=model_name)}.csv")
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def parallel_truncate_by_sentences(self, corpus_loader: ILoadModel) -> Path:
        logger.info("parallel_truncate_by_sentences")
        num_worker = 40

        corpus_len = corpus_loader.load_corpus().total
        max_workers = num_worker

        retriever = LoadRetriever(self.dataset_name, self.model_name).load_retriever()
        tokenizer = retriever.tokenizer

        output_path_base = self.cache_csv_path()
        output_pathes = []
        futures = []

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for i in range(num_worker):
                start_at = i * corpus_len // num_worker
                end_at = min(corpus_len, (i + 1) * corpus_len // num_worker)

                output_path_filename = f"tmp.{output_path_base.stem}.{i}"
                output_path = output_path_base.parent.joinpath(output_path_filename)
                include_header = i == 0
                future = executor.submit(
                    truncate_by_sentences,
                    corpus_loader,
                    start_at,
                    end_at,
                    self.max_doc_len,
                    tokenizer,
                    output_path,
                    include_header,
                )
                output_pathes.append(output_path)
                futures.append(future)

        logger.info("waiting all process completed...")
        for future in as_completed(futures):
            path = future.result()
        logger.info("done all process")

        write_count = util.marge_files(output_pathes, output_path_base)
        logger.info(f"preprocessed corpus length: {write_count}")

        return output_path_base

    def run(self) -> None:
        logger.info(f"self.output(): {self.output()._path()}")

        if self.dataset_name not in AVAILABLE_DATASET:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
        dataset_loader = AVAILABLE_DATASET[self.dataset_name]

        if self.model_name in ["bm25"]:
            logger.info("load ance preprocess result for bm25")
            csv_corpus_path = self.cache_csv_path(model_name="ance")
            if not csv_corpus_path.exists():
                raise Exception("ance csv data does not exist.")
        else:
            csv_corpus_path = self.parallel_truncate_by_sentences(dataset_loader)

        corpus_loader = PolarsCorpusLoader(csv_corpus_path)

        queries = dataset_loader.load_queries()
        if len(queries) > 1000:
            queries = {
                qid: query for i, (qid, query) in enumerate(queries.items()) if i < 1000
            }
        qrels = dataset_loader.load_qrels()

        if corpus_loader is None:
            raise Exception("corpus_loader is None")

        self.dump((corpus_loader, queries, qrels))
