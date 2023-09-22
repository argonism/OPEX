from collections import defaultdict
from logging import getLogger
from typing import Dict, NamedTuple

import ir_datasets
from tqdm import tqdm

from ._base import (
    ILoadModel,
    QrelsDict,
    QueriesDict,
    LargeCorpusSequentialDict,
)

logger = getLogger(__name__)


class LoadMSMarcoPassage(ILoadModel):
    def __init__(self) -> None:
        super().__init__()
        self.download_dataset()

    def download_dataset(self) -> None:
        dataset = ir_datasets.load("msmarco-passage/dev")
        for _ in dataset.docs_iter():
            break

    def load_corpus(self) -> LargeCorpusSequentialDict:
        def preprocess(doc: NamedTuple) -> Dict[str, str]:
            return {"id": doc.doc_id, "text": doc.text}

        dataset = ir_datasets.load("msmarco-passage/dev")

        return LargeCorpusSequentialDict(dataset, preprocess)

    def load_queries(self) -> QueriesDict:
        dataset = ir_datasets.load("msmarco-passage/dev")
        queries = {}
        for query in tqdm(dataset.queries_iter(), total=dataset.queries_count()):
            queries[query.query_id] = query.text

        return queries

    def load_qrels(self) -> QrelsDict:
        dataset = ir_datasets.load("msmarco-passage/dev")
        qrels: Dict[str, Dict[str, int]] = defaultdict(dict)
        for qrel in tqdm(dataset.qrels_iter(), total=dataset.qrels_count()):
            qrels[qrel.query_id][qrel.doc_id] = qrel.relevance

        return qrels
