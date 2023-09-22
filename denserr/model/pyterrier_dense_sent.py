from __future__ import annotations

import os
import pickle
import re
from collections import defaultdict
from logging import getLogger
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, List, Optional, Union

import faiss
import more_itertools
import numpy as np
import pandas as pd
import pyterrier as pt
from pyterrier.model import add_ranks
from pyterrier.transformer import TransformerBase

from denserr.model._base import Retriever
from denserr.utils.util import index_base_path, breakup_to_sentenses, aggregate_sentences

logger = getLogger(__name__)


class PyTDenseIndexer(TransformerBase):
    def __init__(
        self,
        encoder,
        index_path: Path,
        num_docs: Optional[int] = None,
        verbose: bool = True,
        segment_size: int = 500_000,
        **kwargs,
    ) -> None:
        self.index_path = index_path
        self.encoder = encoder
        self.verbose = verbose
        self.num_docs = num_docs
        if self.verbose and self.num_docs is None:
            raise ValueError("if verbose=True, num_docs must be set")
        self.segment_size = segment_size

    def index(self, generator):
        os.makedirs(self.index_path, exist_ok=True)

        docid2docno = []

        def gen_tokenize():
            kwargs = {}
            if self.num_docs is not None:
                kwargs["total"] = self.num_docs
            for doc in (
                pt.tqdm(generator, desc="Indexing", unit="d", **kwargs)
                if self.verbose
                else generator
            ):
                docid2docno.append(doc["docno"])

                yield {"text": doc["text"]}

        segment = -1
        shard_size = []
        for docs in more_itertools.ichunked(gen_tokenize(), self.segment_size):
            segment += 1

            print("Segment %d" % segment)
            docs = list(docs)
            passage_embedding = self.encoder.encode_corpus(
                docs, 16, convert_to_tensor=False
            )

            # if segment <= 0:
            #     from denserr.model.load_model import LoadRetriever

            #     retriever = LoadRetriever("robust04", "ance").load_retriever()
            #     print(f"in indexing shard0 first doc: {docid2docno[0]}")
            #     print(docs[0])
            #     print(passage_embedding[0][:5])
            #     retriever.single_doc_score("test query", docs[0]["text"])

            shard_file_path = self.index_path.joinpath(str(segment) + ".pkl")
            shard_file_path.write_bytes(pickle.dumps(passage_embedding))

            passage_embedding = None

            shard_size.append(len(docs))

        with pt.io.autoopen(os.path.join(self.index_path, "shards.pkl"), "wb") as f:
            pickle.dump(shard_size, f)
            pickle.dump(docid2docno, f)
        return self.index_path


class PyTDenseRetrieval(TransformerBase):
    def __init__(
        self,
        dense_encoder,
        index_path: Path,
        num_results=10000,
        **kwargs,
    ):
        self.num_results = num_results
        self.encoder = dense_encoder
        self.index_path = index_path

    def load_shard_metadata(self) -> None:
        logger.debug("Loading shard metadata")
        shards_files = os.path.join(self.index_path, "shards.pkl")
        with pt.io.autoopen(shards_files) as f:
            self.shard_sizes = pickle.load(f)
            self.docid2docno = pickle.load(f)
        self.segments = len(self.shard_sizes)

    def yield_shard_indexes(
        self, shard_sizes: List[int], index_path: Path
    ) -> Generator[tuple[Any, int], None, None]:
        offset = 0
        for i, shard_size in enumerate(
            pt.tqdm(shard_sizes, desc="Loading shards", unit="shard")
        ):
            shard_path = index_path.joinpath(str(i) + ".pkl")
            passage_embs = pickle.loads(shard_path.read_bytes())

            yield passage_embs, offset

            offset += shard_size

    def __str__(self) -> str:
        return "PyTDenseIndexer"

    def transform(self, topics: pd.DataFrame) -> pd.DataFrame:
        self.load_shard_metadata()

        queries = topics["query"].to_list()
        qid2q = {qid: q for q, qid in zip(queries, topics["qid"].to_list())}

        print("***** inference of %d queries *****" % len(qid2q))
        query_embedding = self.encoder.encode_queries(
            queries, batch_size=16, convert_to_tensor=False
        )

        print(
            "***** faiss search for %d queries on %d shards *****"
            % (len(qid2q), self.segments)
        )
        rtr = []
        indexes_iter = self.yield_shard_indexes(self.shard_sizes, self.index_path)
        for passage_embs, offset in pt.tqdm(indexes_iter, desc="Calc Scores"):
            scores = np.matmul(query_embedding, passage_embs.T)
            sorted_scores = []
            neighbours = []
            for score_list in scores:
                sorted_i_score = sorted(
                    enumerate(score_list), key=lambda x: x[1], reverse=True
                )
                sorted_scores.append([score for _, score in sorted_i_score])
                neighbours.append([i for i, _ in sorted_i_score])

            res = self._calc_scores(
                topics["qid"].values,
                np.array(neighbours),
                np.array(sorted_scores),
                qid2q,
                num_results=self.num_results,
                offset=offset,
            )
            rtr.append(res)
        rtr = pd.concat(rtr)
        rtr = add_ranks(rtr)
        rtr = rtr[rtr["rank"] < self.num_results]
        rtr = rtr.sort_values(
            by=["qid", "score", "docno"], ascending=[True, False, True]
        )
        return rtr

    def _calc_scores(
        self,
        query_embedding2id: List[str],
        I_nearest_neighbor: np.ndarray,
        I_scores: np.ndarray,
        qid2q: Dict[str, str],
        num_results: int = 10000,
        offset: int = 0,
    ) -> pd.DataFrame:
        rtr = []
        # print("docid2docno:", len(self.docid2docno))
        for query_idx in range(I_nearest_neighbor.shape[0]):
            query_id = query_embedding2id[query_idx]

            top_ann_pid = I_nearest_neighbor[query_idx, :].copy()
            scores = I_scores[query_idx, :].copy()
            selected_ann_idx = top_ann_pid[:num_results]
            rank = 0
            seen_pid = set()

            for i, idx in enumerate(selected_ann_idx):
                rank += 1
                # print(f"reffering: {idx + offset}")
                docno = self.docid2docno[idx + offset]
                rtr.append([query_id, qid2q[query_id], idx, docno, rank, scores[i]])
                seen_pid.add(idx)

        return pd.DataFrame(
            rtr, columns=["qid", "query", "docid", "docno", "rank", "score"]
        )


class PyTDenseRetriever(Retriever):
    _instance = None

    def __init__(
        self,
        encoder: Any,
        dataset_name: str,
        model_name: str,
        index_prefix: str = "",
        topk: int = 10000,
        segment_size: int = 500000,
    ) -> None:
        if not pt.started():
            pt.init()

        self.index_path = index_base_path.joinpath(
            index_prefix, model_name, dataset_name
        )
        self.topk = topk
        self.encoder = encoder
        self.tokenizer = encoder.tokenizer

        self.indexer = PyTDenseIndexer(
            encoder, self.index_path, num_docs=topk, segment_size=segment_size
        )
        self.index_pipe = (
            pt.apply.generic(breakup_to_sentenses)
            >> self.indexer
        )

    def indexing(
        self,
        corpus_iter: Iterable,
        indexer: TransformerBase,
        overwrite: bool = False,
        fields: List[str] = ["text", "title"],
    ) -> None:
        index_path = indexer.index_path
        if not overwrite and index_path.joinpath("shards.pkl").exists():
            logger.info(f"shards.pkl found. Use existsing index : {index_path}")
            return None
        logger.info(f"indexing with index_path: {index_path}")
        indexer.index(corpus_iter)

    def query_preprocess(self, query: str) -> str:
        query = query.replace("/", " ")
        query = query.replace("'", " ")
        query = query.replace("\n", " ")
        query = query.replace("?", " ")
        query = query.replace(")", "")
        query = query.replace("(", "")
        query = query.replace(":", "")
        return query

    def preprocess_topics(self, topics: pd.DataFrame) -> pd.DataFrame:
        topics = topics.rename(columns={"title": "query"})
        topics["query"] = topics["query"].map(self.query_preprocess)
        return topics

    def corpus_iter(self, docs: Iterable[dict]) -> Iterable[Dict[str, str]]:
        for doc in docs:
            yield {"docno": doc["id"], "text": doc["text"]}

    def retrieve(
        self,
        corpus: Iterable,
        queries: Dict[str, str],
        **kwargs: Dict[Any, Any],
    ) -> Dict[str, Dict[str, float]]:
        return self.__retrieve(corpus, queries, self.indexer, overwrite=False)

    def __retrieve(
        self,
        corpus: Iterable[dict],
        queries: Dict[str, str],
        indexer: TransformerBase,
        overwrite: bool = False,
    ) -> Dict[str, Dict[str, float]]:
        self.indexing(self.corpus_iter(corpus), indexer, overwrite=overwrite)

        retriever = PyTDenseRetrieval(self.encoder, indexer.index_path)
        topics = pd.DataFrame.from_dict(
            {"qid": queries.keys(), "query": queries.values()}
        )
        topics = self.preprocess_topics(topics)
        result_df = retriever.transform(topics)
        result = aggregate_sentences(result_df)

        # result: defaultdict = defaultdict(dict)
        # for _, row in result_df.iterrows():
        #     result[row["qid"]][row["docno"]] = float(row["score"])
        return result

    def single_doc_score(self, query: str, text: str) -> float:
        query = self.query_preprocess(query)
        q_emb = self.encoder.encode_queries([query], 16, convert_to_tensor=False)
        d_emb = self.encoder.encode_corpus(
            [{"text": text}], 16, convert_to_tensor=False
        )
        score = np.matmul(q_emb, d_emb.T)[0].tolist()[0]
        return float(score)
