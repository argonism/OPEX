from collections import defaultdict
from logging import getLogger
from typing import Dict, List, Tuple, NamedTuple

import ir_datasets
import luigi
import numpy as np
import pandas as pd
import pyterrier as pt
from beir.retrieval.evaluation import EvaluateRetrieval
from gokart.config_params import inherits_config_params
from gokart.target import TargetOnKart
from tqdm import tqdm
from scipy import stats

from denserr.config import DenseErrConfig
from denserr.dataset._base import QrelsDict, QueriesDict
from denserr.dataset.load_dataset import (
    AVAILABLE_DATASET,
    LoadDataset,
)
from denserr.model.load_model import LoadRetriever
from denserr.utils.template import GokartTask

logger = getLogger(__name__)


@inherits_config_params(DenseErrConfig)
class Evaluate(GokartTask):
    dataset_name = luigi.Parameter()
    model_name = luigi.Parameter()

    use_pyterrier = luigi.BoolParameter()

    def requires(self) -> Tuple[GokartTask, GokartTask]:
        return LoadDataset(), Retrieve()

    def output(self) -> TargetOnKart:
        return self.make_target(
            f"denserr/evaluate/{self.dataset_name}/{self.model_name}.pkl"
        )

    def run(self) -> None:
        print(f"cache output path: {self.output()._path()}")
        (_, _, qrels), results = self.load()
        k_values = [10, 100]
        ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(
            qrels, results, k_values
        )
        mrr = EvaluateRetrieval.evaluate_custom(qrels, results, k_values, metric="mrr")
        eval_result = {}
        for eval_metric in [ndcg, _map, recall, precision, mrr]:
            for key, metric in eval_metric.items():
                eval_result[key] = metric

        print(f'||{"|".join(eval_result.keys())}|')
        print("|-" * (len(eval_result) + 1) + "|")
        print(f"|{self.model_name}|" + "|".join(map(str, eval_result.values())) + "|")

        self.dump(eval_result)


@inherits_config_params(DenseErrConfig)
class EvaluatePerQuery(GokartTask):
    dataset_name = luigi.Parameter()
    model_name = luigi.Parameter()

    use_pyterrier = luigi.BoolParameter()

    def requires(self) -> Tuple[GokartTask, GokartTask]:
        return LoadDataset(), Retrieve()

    def output(self) -> TargetOnKart:
        return self.make_target(
            f"denserr/evaluatePerQuery/{self.dataset_name}/{self.model_name}.pkl"
        )

    def run(self) -> None:
        import pytrec_eval

        print(f"cache output path: {self.output()._path()}")
        (_, _, qrels), results = self.load()

        for qid, rels in results.items():
            for pid in list(rels):
                if qid == pid:
                    results[qid].pop(pid)

        ndcg = {}
        _map = {}
        recall = {}
        precision = {}

        k_values = [10, 100]
        for k in k_values:
            ndcg[f"NDCG@{k}"] = []
            _map[f"MAP@{k}"] = []
            recall[f"Recall@{k}"] = []
            precision[f"P@{k}"] = []

        map_string = "map_cut." + ",".join([str(k) for k in k_values])
        ndcg_string = "ndcg_cut." + ",".join([str(k) for k in k_values])
        recall_string = "recall." + ",".join([str(k) for k in k_values])
        precision_string = "P." + ",".join([str(k) for k in k_values])
        evaluator = pytrec_eval.RelevanceEvaluator(
            qrels, {map_string, ndcg_string, recall_string, precision_string}
        )
        scores = evaluator.evaluate(results)

        for query_id in scores.keys():
            for k in k_values:
                ndcg[f"NDCG@{k}"].append(scores[query_id]["ndcg_cut_" + str(k)])
                _map[f"MAP@{k}"].append(scores[query_id]["map_cut_" + str(k)])
                recall[f"Recall@{k}"].append(scores[query_id]["recall_" + str(k)])
                precision[f"P@{k}"].append(scores[query_id]["P_" + str(k)])

        result = {
            metric_name: values
            for metric in [ndcg, _map, recall, precision]
            for metric_name, values in metric.items()
        }

        self.dump(result)


@inherits_config_params(DenseErrConfig)
class Retrieve(GokartTask):
    dataset_name = luigi.Parameter()
    model_name = luigi.Parameter()

    use_pyterrier = luigi.BoolParameter()

    def requires(self) -> GokartTask:
        return LoadDataset()

    def output(self) -> TargetOnKart:
        return self.make_target(
            f"denserr/retrieve/{self.dataset_name}/{self.model_name}.pkl"
        )

    def run(self) -> None:
        print(f"cache output path: {self.output()._path()}")
        corpus, queries, _ = self.load()
        logger.info(f"queries len: {len(queries)}")
        retriever = LoadRetriever(self.dataset_name, self.model_name).load_retriever()

        results = retriever.retrieve(corpus, queries)
        logger.info(f"Retrieve (with {self.model_name}): done retrieving")
        qids = list(results.keys())
        print("len(results):", len(results))
        print("ranking len:", len(results[qids[0]]))

        self.dump((results))


@inherits_config_params(DenseErrConfig)
class RetrieveForDebug(GokartTask):
    dataset_name = luigi.Parameter()
    model_name = luigi.Parameter()

    def requires(self) -> Tuple[GokartTask, GokartTask]:
        return LoadDataset()

    def output(self) -> TargetOnKart:
        return self.make_target(
            f"denserr/retrieve/{self.dataset_name}/{self.model_name}.pkl"
        )

    def run(self) -> None:
        corpus, queries, _ = self.load()

        corpus.frame = corpus.frame.limit(100)
        queries = {
            qid: query for i, (qid, query) in enumerate(queries.items()) if i < 10
        }
        qids = list(queries.keys())

        logger.info(f"corpus len: {len(corpus)}")
        logger.info(f"queries len: {len(queries)}")
        retriever = LoadRetriever(self.dataset_name, self.model_name).load_retriever()
        results = retriever.retrieve(corpus, queries)
        print(f"len(results): {len(results)}")
        print(f"len(results[qids[0]]): {len(results[qids[0]])}")

        for qid in results:
            for docid, score in results[qid].items():
                query = queries[qid]
                doc = corpus[docid]
                print(doc)
                re_score = retriever.single_doc_score(
                    query,
                    doc["text"],
                )
                print(score)
                print(float(re_score))


pyserini_indexes = {"msmarco-pas": {"ance": "msmarco-passage-ance-bf"}}


@inherits_config_params(DenseErrConfig)
class RetrieveWithPyserini(GokartTask):
    dataset_name = luigi.Parameter()
    model_name = luigi.Parameter()

    def output(self) -> TargetOnKart:
        return self.make_target(
            f"denserr/retrieve_pyserini/{self.dataset_name}/{self.model_name}.pkl"
        )

    def run(self) -> None:
        from pyserini.search.faiss import AnceEncoder, AnceQueryEncoder, FaissSearcher

        if self.dataset_name not in AVAILABLE_DATASET:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
        dataset_loader = AVAILABLE_DATASET[self.dataset_name]
        queries = dataset_loader.load_queries()

        query_encoder = AnceQueryEncoder(
            "castorini/ance-msmarco-passage", device="cuda:0"
        )
        searcher = FaissSearcher.from_prebuilt_index(
            "msmarco-passage-ance-bf", query_encoder
        )

        results: Dict[str, Dict[str, float]] = defaultdict(dict)
        for i, (qid, query) in enumerate(tqdm(queries.items())):
            if i >= 1000:
                break
            hits = searcher.search(query, k=10000, threads=32)
            for i in range(len(hits)):
                results[qid][hits[i].docid] = hits[i].score

        self.dump((results))


@inherits_config_params(DenseErrConfig)
class Rerank(GokartTask):
    dataset_name = luigi.Parameter()
    model_name = luigi.Parameter()
    with_pyserini = luigi.BoolParameter()

    def requires(self) -> Tuple[GokartTask, GokartTask]:
        if self.with_pyserini:
            retrieve_task = RetrieveWithPyserini()
        else:
            retrieve_task = Retrieve()
        return retrieve_task, PreprocessedDataset()

    def output(self) -> TargetOnKart:
        return self.make_target(
            f"denserr/rerank/{self.dataset_name}/{self.model_name}.pkl"
        )

    def run(self) -> None:
        results, (corpus, queries, _) = self.load()
        retriever = LoadRetriever(self.dataset_name, self.model_name).load_retriever()

        rerank_results = {}
        for qid in tqdm(queries):
            tmp_queries = {qid: queries[qid], "psudo": "psudo query"}
            tmp_corpus = {did: corpus[did] for did in results[qid]}
            rerank_result = retriever.retrieve(tmp_corpus, tmp_queries)
            rerank_results[qid] = rerank_result[qid]

        self.dump((rerank_results))


@inherits_config_params(DenseErrConfig)
class TuningBM25(GokartTask):
    dataset_name = luigi.Parameter()
    model_name = luigi.Parameter()
    metric = luigi.Parameter()

    use_pyterrier = luigi.BoolParameter()

    def requires(self) -> GokartTask:
        return Retrieve(), LoadDataset()

    def output(self) -> TargetOnKart:
        return self.make_target(
            f"denserr/tuning/{self.dataset_name}/bm25/{self.metric}.pkl"
        )

    def run(self) -> None:
        print(f"cache output path: {self.output()._path()}")
        results, (corpus, queries, qrels) = self.load()

        logger.info(f"queries len: {len(queries)}")
        if not self.model_name == "bm25":
            logger.warning(
                f"current model_name is {self.model_name} but loading bm25 ignore model_name."
            )

        retriever = LoadRetriever(self.dataset_name, "bm25").load_retriever()
        bm25 = retriever.get_bm25()
        topics = pd.DataFrame.from_dict(
            {"qid": queries.keys(), "query": queries.values()}
        )
        topics = retriever.preprocess_topics(topics)

        qrels_records = []
        for qid in qrels:
            for docno, label in qrels[qid].items():
                qrels_records.append((qid, docno, label))
        qrels_df = pd.DataFrame.from_records(
            qrels_records, columns=["qid", "docno", "label"]
        )

        b_range = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        k1_range = [0.3, 0.6, 0.9, 1.2, 1.4, 1.6, 2]

        logger.info(f"b_range: {b_range}")
        logger.info(f"k1_range: {k1_range}")
        logger.info("start bm25 grid search")

        result = pt.GridSearch(
            bm25,
            {bm25: {"bm25.b": b_range, "bm25.k_1": k1_range}},
            topics,
            qrels_df,
            self.metric,
            jobs=20,
            verbose=True,
        )
        logger.info(f"done grid search: {result}")
        self.dump(result)

        # pad = 1e-3
        # b_range = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        # k1_range = [0.3, 0.6, 0.9, 1.2, 1.4, 1.6, 2]
        # k3_range = [0.5, 2, 4, 6, 8, 10, 12, 14, 20]

        # logger.info(f"b_range: {b_range}")
        # logger.info(f"k1_range: {k1_range}")
        # logger.info("start bm25 grid search")

        # result = pt.GridSearch(
        #     bm25,
        #     {bm25: {"bm25.b": b_range, "bm25.k_1": k1_range, "bm25.k_3": k3_range}},
        #     topics,
        #     qrels_df,
        #     self.metric,
        #     verbose=True,
        # )
        # logger.info(f"done grid search: {result}")
        # self.dump(result)


@inherits_config_params(DenseErrConfig)
class TuningBM25WithMSMARCODoc(GokartTask):
    dataset_name = luigi.Parameter()
    model_name = luigi.Parameter()
    metric = luigi.Parameter()

    use_pyterrier = luigi.BoolParameter()

    def requires(self) -> Tuple[GokartTask, GokartTask]:
        return LoadDataset()

    def output(self) -> TargetOnKart:
        return self.make_target(
            f"denserr/tuning/{self.dataset_name}/bm25/{self.metric}.pkl"
        )

    def load_train_queries_qrels(
        self, dataset_key: str = "msmarco-document/train", queries_num: int = 250
    ) -> Tuple[QueriesDict, QrelsDict]:
        logger.info(f"loading train queries and qrels from {dataset_key}")
        dataset = ir_datasets.load(dataset_key)

        queries = {}
        for i, query in enumerate(
            tqdm(dataset.queries_iter(), total=dataset.queries_count())
        ):
            queries[query.query_id] = query.text
            if i >= queries_num:
                break

        qrels: Dict[str, Dict[str, int]] = defaultdict(dict)
        for qrel in tqdm(dataset.qrels_iter(), total=dataset.qrels_count()):
            qrels[qrel.query_id][qrel.doc_id] = qrel.relevance

        return queries, qrels

    def run(self) -> None:
        print(f"cache output path: {self.output()._path()}")
        corpus, _, _ = self.load()

        dataset_key = "msmarco-document/train"
        queries, qrels = self.load_train_queries_qrels(dataset_key)

        logger.info(f"queries len: {len(queries)}")
        if not self.model_name == "bm25":
            logger.warning(
                f"current model_name is {self.model_name} but loading bm25 ignore model_name."
            )

        retriever = LoadRetriever("msmarco-doc", "bm25").load_retriever()
        bm25 = retriever.get_bm25()

        topics = pd.DataFrame.from_dict(
            {"qid": queries.keys(), "query": queries.values()}
        )
        topics = retriever.preprocess_topics(topics)

        qrels_records = []
        for qid in qrels:
            for docno, label in qrels[qid].items():
                qrels_records.append((qid, docno, label))
        qrels_df = pd.DataFrame.from_records(
            qrels_records, columns=["qid", "docno", "label"]
        )

        # b_range = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        # k1_range = [0.3, 0.6, 0.9, 1.2, 1.4, 1.6, 2]
        b_range = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        k1_range = [0.4, 0.8, 1.2, 1.6, 2.0, 2.4, 2.8, 3.2, 3.6, 4]

        logger.info(f"b_range: {b_range}")
        logger.info(f"k1_range: {k1_range}")
        logger.info("start bm25 grid search")

        result = pt.GridSearch(
            bm25,
            {bm25: {"bm25.b": b_range, "bm25.k_1": k1_range}},
            topics,
            qrels_df,
            self.metric,
            jobs=40,
            verbose=True,
        )
        logger.info(f"done grid search: {result}")
        self.dump(result)
