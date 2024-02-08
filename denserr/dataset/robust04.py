from collections import defaultdict
from logging import getLogger
from typing import NamedTuple, Dict

import ir_datasets
from tqdm import tqdm

from ._base import (
    CorpusDict,
    ILoadModel,
    QrelsDict,
    QueriesDict,
    LargeCorpusSequentialDict,
)

logger = getLogger(__name__)


class LoadRobust04(ILoadModel):
    def __init__(self) -> None:
        super().__init__()
        self.dataset_key = "disks45/nocr/trec-robust-2004"
        self.download_dataset()

    def download_dataset(self) -> None:
        dataset = ir_datasets.load(self.dataset_key)
        for _ in dataset.docs_iter():
            break

    def load_corpus(self) -> LargeCorpusSequentialDict:
        def preprocess(doc: NamedTuple) -> Dict[str, str]:
            return {"id": doc.doc_id, "text": doc.title + " " + doc.body}

        dataset = ir_datasets.load(self.dataset_key)

        return LargeCorpusSequentialDict(dataset, preprocess)

    def load_queries(self) -> QueriesDict:
        dataset = ir_datasets.load(self.dataset_key)
        queries = {}
        for query in tqdm(dataset.queries_iter(), total=dataset.queries_count()):
            queries[query.query_id] = query.description

        return queries

    def load_qrels(self) -> QrelsDict:
        dataset = ir_datasets.load(self.dataset_key)
        qrels: Dict[str, Dict[str, int]] = defaultdict(dict)
        for qrel in tqdm(dataset.qrels_iter(), total=dataset.qrels_count()):
            qrels[qrel.query_id][qrel.doc_id] = qrel.relevance

        return qrels
