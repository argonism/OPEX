from __future__ import annotations

import multiprocessing as mp
import os
import pickle
import re
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from logging import getLogger
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, List, Optional, Union, Tuple

from tqdm import tqdm
import torch
import faiss
import more_itertools
import numpy as np
import pandas as pd
import pyterrier as pt
from pyterrier.model import add_ranks
from pyterrier.transformer import TransformerBase

from denserr.model._base import Retriever
from denserr.utils.util import index_base_path, IterCacher, cache_dir

logger = getLogger(__name__)


def index_with_multiprocessing(
    docs_iter, encoder_loader, model_confs, batch_size, device, order: int
):
    logger.info(f"index_with_multiprocessing of order {order}, {device} from logger")
    print(f"index_with_multiprocessing of order {order}, {device}")
    docs = list(docs_iter)
    encoder = encoder_loader(device, **model_confs)
    passage_embedding = encoder.encode_corpus(docs, batch_size, convert_to_tensor=False)
    return (order, passage_embedding)


class PyTParallelDenseIndexer(TransformerBase):
    def __init__(
        self,
        load_ance,
        index_path: Path,
        model_confs: Dict[str, str] = {},
        num_docs: Optional[int] = None,
        verbose: bool = True,
        segment_size: int = 1_000_000,
        **kwargs,
    ) -> None:
        self.index_path = index_path
        self.model_confs = model_confs
        self.load_ance = load_ance
        self.verbose = verbose
        self.num_docs = num_docs
        if self.verbose and self.num_docs is None:
            raise ValueError("if verbose=True, num_docs must be set")
        self.segment_size = segment_size
        self.max_workers = 4
        self.batch_size = 16

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

        segment = 0
        shard_size = []
        for docs in tqdm(more_itertools.ichunked(gen_tokenize(), self.segment_size)):
            shard_file_path = self.index_path.joinpath(str(segment) + ".pkl")
            if shard_file_path.exists():
                logger.info(f"index segment {segment} shard found {shard_file_path}")
                logger.info("skip encoding")
                continue
            with ProcessPoolExecutor(
                max_workers=self.max_workers, mp_context=mp.get_context("spawn")
            ) as executor:
                print("Segment %d" % segment)
                docs = list(docs)

                futures = []
                for i, docs_iter in enumerate(
                    more_itertools.divide(self.max_workers, docs)
                ):
                    device = f"cuda:{i}"

                    future = executor.submit(
                        index_with_multiprocessing,
                        docs_iter,
                        self.load_ance,
                        self.model_confs,
                        self.batch_size,
                        device,
                        i,
                    )
                    logger.info(f"task at {i} is queued")
                    futures.append(future)

                logger.info("waiting all process completed...")
                results = []
                for future in tqdm(
                    as_completed(futures), desc="waiting futures", total=len(futures)
                ):
                    order, passage_embedding_chunk = future.result()
                    results.append((order, passage_embedding_chunk))

                embedding_chunks = []
                for order, passage_embedding_chunk in sorted(
                    results, key=lambda x: x[0]
                ):
                    embedding_chunks.append(passage_embedding_chunk)

                passage_embeddings = np.concatenate(embedding_chunks, axis=0)

                shard_file_path.write_bytes(pickle.dumps(passage_embeddings))
                passage_embedding = None

                shard_size.append(len(docs))
            segment += 1

        with pt.io.autoopen(os.path.join(self.index_path, "shards.pkl"), "wb") as f:
            pickle.dump(shard_size, f)
            pickle.dump(docid2docno, f)
        return self.index_path


def _calc_scores(
    docid2docno,
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
            docno = docid2docno[idx + offset]
            rtr.append([query_id, qid2q[query_id], idx, docno, rank, scores[i]])
            seen_pid.add(idx)

    return pd.DataFrame(
        rtr, columns=["qid", "query", "docid", "docno", "rank", "score"]
    )


def retrieve_in_parallel(
    query_embedding,
    passage_embs,
    topics,
    qid2q,
    num_results,
    offset,
    docid2docno,
    i,
    with_faiss: bool = False,
):
    def calc_scores_with_faiss(
        psg_embs: np.ndarray, query_embs: np.ndarray, num_results
    ) -> tuple[np.ndarray, np.ndarray]:
        dim = psg_embs.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(psg_embs)

        faiss.omp_set_num_threads(16)
        scores, neighbours = index.search(query_embs, num_results)
        return scores, neighbours

    print(f"retrieve_in_parallel with offset {offset}")
    if with_faiss:
        print("retrieving with faiss")
        sorted_scores, neighbours = calc_scores_with_faiss(
            passage_embs, query_embedding, num_results
        )
    else:
        query_embedding = torch.from_numpy(query_embedding).to(f"cuda:{i}")
        passage_embs = torch.from_numpy(passage_embs).to(f"cuda:{i}")
        scores = torch.matmul(query_embedding, passage_embs.T).cpu().numpy()
        # scores = np.matmul(query_embedding, passage_embs.T)
        sorted_scores = []
        neighbours = []
        for score_list in scores:
            sorted_i_score = sorted(
                enumerate(score_list), key=lambda x: x[1], reverse=True
            )
            sorted_scores.append([score for _, score in sorted_i_score])
            neighbours.append([i for i, _ in sorted_i_score])

    res = _calc_scores(
        docid2docno,
        topics["qid"].values,
        np.array(neighbours),
        np.array(sorted_scores),
        qid2q,
        num_results=num_results,
        offset=offset,
    )
    return res


class PyTParallelDenseRetrieval(TransformerBase):
    def __init__(
        self,
        encoder,
        index_path: Path,
        dataset_name: str,
        model_name: str,
        num_results=10000,
        **kwargs,
    ):
        self.num_results = num_results
        self.encoder = encoder
        self.index_path = index_path
        self.max_workers = 4
        self.dataset_name = dataset_name
        self.model_name = model_name

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
        rtr = pd.DataFrame()
        indexes_iter = self.yield_shard_indexes(self.shard_sizes, self.index_path)
        cache_path = cache_dir.joinpath(
            "pyt_denser_parallel",
            f"{self.dataset_name}_{self.model_name}_{self.num_results}",
        )
        iter_cacher = IterCacher(
            cache_path, cache_strategy="each_file", key_gen_method="hash"
        )
        for cacher, (passage_embs, offset), result in iter_cacher.iter(
            pt.tqdm(indexes_iter, desc="Calc Scores")
        ):
            if result is None:
                with ProcessPoolExecutor(
                    max_workers=self.max_workers, mp_context=mp.get_context("spawn")
                ) as executor:
                    futures = []
                    for i, emb_index in enumerate(
                        more_itertools.divide(
                            self.max_workers, range(len(passage_embs))
                        )
                    ):
                        passage_embs_part = passage_embs[list(emb_index)]
                        print("passage_embs_part len:", len(passage_embs_part))
                        logger.info(f"queing task with offset: {offset}")
                        future = executor.submit(
                            retrieve_in_parallel,
                            query_embedding,
                            passage_embs_part,
                            topics,
                            qid2q,
                            self.num_results,
                            offset,
                            self.docid2docno,
                            i,
                            with_faiss=False,
                        )
                        futures.append(future)
                        offset = offset + len(passage_embs_part)

                    dfs = []
                    logger.info("waiting all process completed...")
                    for future in tqdm(as_completed(futures)):
                        res = future.result()
                        dfs.append(res)

                    result = pd.concat(dfs)

            rtr = pd.concat([rtr, result])
            rtr = add_ranks(rtr)
            rtr = rtr[rtr["rank"] < self.num_results]
            rtr = rtr.drop("rank", axis=1)
            cacher.cache(rtr)
            print("current rtr len:", len(rtr))

        print("forming final result")
        rtr = add_ranks(rtr)
        rtr = rtr[rtr["rank"] < self.num_results]
        rtr = rtr.sort_values(
            by=["qid", "score", "docno"], ascending=[True, False, True]
        )
        return rtr


class PyTParallelDenseRetriever(Retriever):
    _instance = None

    def __init__(
        self,
        load_ance: Any,
        dataset_name: str,
        model_name: str,
        model_confs: Dict[str, str] = {},
        index_prefix: str = "",
        topk: int = 10000,
        segment_size: int = 500000,
        overwrite: bool = False,
    ) -> None:
        if not pt.started():
            pt.init()

        self.index_path = index_base_path.joinpath(
            index_prefix, model_name, dataset_name
        )
        self.topk = topk
        self.overwrite = overwrite

        self.indexer = PyTParallelDenseIndexer(
            load_ance,
            self.index_path,
            model_confs=model_confs,
            num_docs=topk,
            segment_size=segment_size,
        )
        self.encoder = load_ance("cuda:0", **model_confs)
        self.retriever = PyTParallelDenseRetrieval(
            self.encoder,
            self.index_path,
            dataset_name,
            model_name,
            num_results=self.topk,
        )
        self.tokenizer = self.encoder.tokenizer

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
        return self.__retrieve(corpus, queries, self.indexer, overwrite=self.overwrite)

    def __retrieve(
        self,
        corpus: Iterable[dict],
        queries: Dict[str, str],
        indexer: TransformerBase,
        overwrite: bool = False,
    ) -> Dict[str, Dict[str, float]]:
        self.indexing(self.corpus_iter(corpus), indexer, overwrite=overwrite)

        topics = pd.DataFrame.from_dict(
            {"qid": queries.keys(), "query": queries.values()}
        )
        topics = self.preprocess_topics(topics)
        result_df = self.retriever.transform(topics)
        print(f"result_df len: {len(result_df)}")
        result: defaultdict = defaultdict(dict)
        for _, row in tqdm(result_df.iterrows()):
            result[row["qid"]][row["docno"]] = float(row["score"])
        print(f"result len: {len(result)}")
        print(f"result[row['qid']] len: {len(result[row['qid']])}")
        return result

    def single_doc_score(self, query: str, text: str) -> float:
        query = self.query_preprocess(query)
        q_emb = self.encoder.encode_queries(
            [query], 16, convert_to_tensor=False, verbose=False
        )
        d_emb = self.encoder.encode_corpus(
            [{"text": text}], 16, convert_to_tensor=False, verbose=False
        )
        score = np.matmul(q_emb, d_emb.T)[0].tolist()[0]
        return float(score)
