import datetime
import logging
import random
from typing import Dict, List, Tuple

import luigi
import nltk
from gokart.config_params import inherits_config_params
from gokart.target import TargetOnKart
from tqdm import tqdm

from denserr.config import DenseErrConfig
from denserr.dataset.load_dataset import LoadDataset
from denserr.model.load_model import DenseRetriever, LoadRetriever
from denserr.retrieve import Rerank, Retrieve
from denserr.utils.template import GokartTask

logger = logging.getLogger(__name__)


def debug_dataset_name(dataset_name: str) -> str:
    return f"debug/{dataset_name}"


def estimate_rank_pos(target_score: float, ranking: List[Tuple[str, float]]) -> int:
    for i, (_, i_score) in enumerate(ranking):
        if target_score > i_score:
            return i + 1
    return len(ranking)


@inherits_config_params(DenseErrConfig)
class DebugSingleScoringRetrieval(GokartTask):
    dataset_name = luigi.Parameter()
    model_name = luigi.Parameter()

    def requires(self) -> Tuple[GokartTask, GokartTask]:
        return LoadDataset()

    def output(self) -> TargetOnKart:
        return self.make_target(
            f"denserr/debug/retrieve/{self.dataset_name}/{self.model_name}.pkl"
        )

    def run(self) -> None:
        corpus, queries, _ = self.load()

        corpus = [doc for i, doc in enumerate(corpus) if i < 100]
        logger.info(f"corpus length: {len(corpus)}")

        # if len(corpus) < 20:
        #     for doc in corpus:
        #         print(doc["id"])
        #         print(doc["text"])
        #         print("")

        # queries = {
        #     qid: query for i, (qid, query) in enumerate(queries.items()) if i < 10
        # }

        retriever = LoadRetriever(self.dataset_name, self.model_name).load_retriever(
            debug=True
        )
        logger.info(f"index_path: {retriever.index_path.joinpath('shards.pkl')}")
        retriever.index_path.joinpath("shards.pkl").unlink(missing_ok=True)
        results = retriever.retrieve(corpus, queries)
        self.dump(results)


@inherits_config_params(DenseErrConfig)
class DebugSingleScoring(GokartTask):
    dataset_name = luigi.Parameter()
    model_name = luigi.Parameter()
    with_pyserini = luigi.BoolParameter()

    def requires(self) -> Tuple[GokartTask, GokartTask]:
        return DebugSingleScoringRetrieval(rerun=self.rerun), LoadDataset()

    def run(self) -> None:
        logging.basicConfig(level=logging.DEBUG)

        results, (corpus_loader, queries, _) = self.load()

        retriever = LoadRetriever(self.dataset_name, self.model_name).load_retriever(
            debug=True
        )

        qids = list(results.keys())
        logger.debug(f"qids: {qids}")
        logger.debug(f"len qids: {len(qids)}")

        corpus = corpus_loader.to_dict()

        subts = []
        rank_shifts = []
        for i, qid in enumerate(tqdm(qids)):
            # a = retriever.index_path.joinpath("data.properties")
            # if a.exists(): a.unlink()
            query = queries[qid]
            logger.debug(f"query: {query} ({qid})")
            at_k = 10

            ranking = sorted(results[qid].items(), key=lambda x: x[1], reverse=True)
            print(f"ranking len: {len(ranking)}")
            if len(ranking) <= at_k:
                print("no ranking result. skipped.")
                continue

            target_doc_id, original_score = ranking[at_k - 1]
            target_doc_text = corpus[target_doc_id]["text"]
            perturbed_score = retriever.single_doc_score(query, target_doc_text)
            subt = abs(original_score - perturbed_score)
            new_rank = estimate_rank_pos(perturbed_score, ranking)
            rank_shifts.append((at_k, new_rank + 1, new_rank - at_k))
            print(f"original_score   : {original_score}")
            print(f"single_doc_score : {perturbed_score}\n")
            print(f"subt             : {subt}\n")
            print(f"rank shift       : {at_k} => {(new_rank)} ({(new_rank) - at_k})\n")
            if subt > 0:
                print(f"query   : {query}")
                print(target_doc_id)
                print(target_doc_text)
                print("")

            subts.append(subt)
            if i >= 100:
                break
        print("ave subts:", sum(subts) / len(subts), "\n")
        print(
            "ave rank_shift:",
            sum([shift for _, _, shift in rank_shifts]) / len(rank_shifts),
            "\n",
        )


@inherits_config_params(DenseErrConfig)
class DebugBatchSingleScoring(GokartTask):
    dataset_name = luigi.Parameter()
    model_name = luigi.Parameter()
    with_pyserini = luigi.BoolParameter()

    def requires(self) -> Tuple[GokartTask, GokartTask]:
        return DebugSingleScoringRetrieval(rerun=self.rerun), LoadDataset()

    def run(self) -> None:
        results, (corpus_loader, queries, _) = self.load()

        retriever = LoadRetriever(self.dataset_name, self.model_name).load_retriever(
            debug=True
        )

        qids = list(results.keys())
        logger.debug(f"qids: {qids}")
        logger.debug(f"len qids: {len(qids)}")

        batch_num = 5
        sample_num = 20

        corpus = corpus_loader.to_dict()
        subts = []
        for i in tqdm(range(sample_num), total=sample_num):
            qid = qids[i]
            query = queries[qid]
            batch_queries = [query] * batch_num
            logger.debug(f"query: {query} ({qid})")

            ranking = sorted(results[qid].items(), key=lambda x: x[1], reverse=True)
            logger.debug(f"ranking len: {len(ranking)}")
            if len(ranking) <= 0:
                logger.warning("no ranking result. skipped.")
                continue

            docs = []
            scores = []
            for target_doc_id, original_score in ranking[:batch_num]:
                target_doc_text = corpus[target_doc_id]["text"]
                docs.append(target_doc_text)
                scores.append(original_score)

            re_scores = retriever.batch_single_doc_score(batch_queries, docs)
            print(re_scores)
            for orig_score, re_score in zip(scores, re_scores):
                subt = abs(orig_score - re_score)
                print(f"original_score   : {orig_score}")
                print(f"single_doc_score : {re_score}\n")
                print(f"subt             : {subt}\n")
                if subt > 2:
                    print(target_doc_id)
                    print(target_doc_text)
                    print("")
                subts.append(subt)
            if i >= sample_num:
                break

        print("ave subts:", sum(subts) / len(subts), "\n")
