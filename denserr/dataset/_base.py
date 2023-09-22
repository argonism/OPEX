import abc
from typing import Iterable, NamedTuple, Callable, Iterator, Dict, Generator, Set, Optional
from pathlib import Path
import json
import logging

from ir_datasets.datasets.base import Dataset
import polars as pl
from tqdm import tqdm

logger = logging.getLogger(__name__)

CorpusDict = Dict[str, Dict[str, str]]
QueriesDict = Dict[str, str]
QrelsDict = Dict[str, Dict[str, int]]

class LargeCorpusSequentialDict(Iterable):
    def __init__(
        self, dataset: Dataset, doc_preprocess: Callable[[NamedTuple], Dict[str, str]]
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.doc_store = self.dataset.docs_store()
        self.doc_preprocess = doc_preprocess
        self.total = dataset.docs_count()

    def __iter__(self) -> Iterator:
        for doc in self.dataset.docs_iter():
            yield self.doc_preprocess(doc)

    def __getitem__(self, key: str) -> dict:
        doc_namedtpl: NamedTuple = self.doc_store.get(key)
        doc = doc_namedtpl._asdict()
        return doc


class ILoadModel(abc.ABC):
    def load_corpus(self) -> LargeCorpusSequentialDict:
        raise NotImplementedError()

    def load_queries(self) -> QueriesDict:
        raise NotImplementedError()

    def load_qrels(self) -> QrelsDict:
        raise NotImplementedError()


class PolarsCorpusLoader(Iterable):
    def __init__(self, corpus_path: Path, batch_size: int = 500_000) -> None:
        self.corpus_path = corpus_path
        logger.info(f"Loading preprocessed corpus from {self.corpus_path}")
        self.frame = pl.scan_csv(corpus_path)
        self.total_rows: Optional[int] = None
        self.batch_size = batch_size

    @property
    def total(self) -> int:
        if self.total_rows is None:
            self.total_rows = self.frame.select(pl.count()).collect(streaming=True)[
                "count"
            ][0]
        return self.total_rows

    def __len__(self) -> int:
        return self.total

    def __iter__(self) -> Generator[dict, None, None]:
        for doc in tqdm(self.frame.collect().iter_rows(named=True)):
            doc["text"] = json.loads(doc["text"])
            yield doc

    def to_dict(self) -> Dict[str, Dict[str, str]]:
        print("self.corpus_path:", self.corpus_path)
        df = self.frame.collect()
        texts = [{"text": json.loads(text)} for text in df["text"]]
        docs = dict(zip(df["id"], texts))
        return docs

    def fetch_docs(self, keys: Set[str]) -> Dict[str, Dict[str, str]]:
        df = self.frame.filter(pl.col("id").is_in(keys)).collect(streaming=True)
        decoded_texts = df["text"].apply(lambda x: json.loads(x))
        docs = dict(zip(df["id"], decoded_texts))
        return docs

    def __getitem__(self, key: str) -> dict:
        item = (
            self.frame.filter(pl.col("id") == key)
            .select("text")
            .collect(streaming=True)
        )
        doc = item.to_dict()
        doc["text"] = json.loads(doc["text"][0])
        return doc
