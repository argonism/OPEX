from typing import Any, Dict, Iterable, List
import abc

import numpy as np
from beir.retrieval.evaluation import EvaluateRetrieval
from denserr.dataset._base import PolarsCorpusLoader


class Retriever(abc.ABC):
    def retrieve(
        self,
        corpus: Iterable,
        queries: Dict[str, str],
        **kwargs: Dict[Any, Any],
    ) -> Dict[str, Dict[str, float]]:
        raise NotImplementedError

    def single_doc_score(self, query: str, text: str) -> float:
        raise NotImplementedError


class DenseRetriever(Retriever):
    def __init__(self, retriever: EvaluateRetrieval):
        self.retriever = retriever
        self.encoder = self.retriever.retriever.model
        self.batch_size = 32
        self.tokenizer = self.encoder.tokenizer

    def single_doc_score(self, query: str, text: str, title: str = "") -> float:
        # self.clear_index()
        corpus = {
            "docid": {"text": text, "title": title},
            "psudo_doc": {"text": "psudo text", "title": "psudo title"},
        }
        queries = {"qid": query, "psudo": ""}
        result = self.retrieve(corpus, queries)
        score = result["qid"]["docid"]
        return score

    def embed_queries(self, queries: List[str]) -> np.ndarray:
        query_embedding: np.ndarray = self.encoder.encode_queries(
            queries,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_tensor=False,
        )
        return query_embedding

    def embed_docs(
        self,
        docs: List[Dict[str, str]],
    ) -> np.ndarray:
        doc_embedding: np.ndarray = self.encoder.encode_corpus(
            docs,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_tensor=False,
        )
        return doc_embedding

    def retrieve(
        self,
        corpus: PolarsCorpusLoader,
        queries: Dict[str, str],
        **kwargs: Dict[Any, Any],
    ) -> Dict[str, Dict[str, float]]:
        results: Dict[str, Dict[str, float]] = self.retriever.retrieve(
            corpus, queries, **kwargs
        )
        return results

    # def load_index(self) -> None:
    #     if self.index_path.exists():
    #         self.retriever.retriever.load(
    #             input_dir=self.index_dir, prefix=self.index_name, ext=self.index_type
    #         )

    # def save_index(self) -> None:
    #     if not self.index_path.exists():
    #         self.retriever.retriever.save(
    #             output_dir=self.index_dir, prefix=self.index_name, ext=self.index_type
    #         )

    # def clear_index(self) -> None:
    #     self.retriever.retriever.faiss_index = None
