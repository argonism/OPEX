import datetime
import logging
import math
import random
from logging import getLogger
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple, Union

import luigi
import matplotlib.pyplot as plt
import nltk
import numpy as np
from gokart.config_params import inherits_config_params
from gokart.target import TargetOnKart
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from denserr.analyzer.distribution_visualizer import DistributionVisualizer
from denserr.config import DenseErrConfig
from denserr.dataset.load_dataset import LoadDataset
from denserr.model.load_model import DenseRetriever, LoadRetriever
from denserr.retrieve import Rerank, Retrieve
from denserr.utils.template import GokartTask
from denserr.utils.util import (
    now_log_dir,
    project_dir,
    writeout_json_to_log,
    cache_dir,
    IterCacher,
)

logger = getLogger(__name__)


@inherits_config_params(DenseErrConfig)
class DamagedAnalyze(GokartTask):
    dataset_name = luigi.Parameter()
    model_name = luigi.Parameter()
    with_pyserini = luigi.BoolParameter()

    sample_repeat_times = luigi.IntParameter(100)
    target_doc_rank = luigi.IntParameter(1)

    damaged_start_at = luigi.IntParameter()
    damaged_until = luigi.IntParameter()

    def estimate_rank_pos(
        self, target_score: float, ranking: List[Tuple[str, float]]
    ) -> int:
        for i, (_, i_score) in enumerate(ranking):
            if target_score > i_score:
                return i + 1
        return len(ranking)

    def log_name(self) -> str:
        return (
            f"{self.dataset_name}_{self.model_name}_@{self.target_doc_rank}_"
            f"{self.damaged_start_at}-{self.damaged_until}_{self.sample_repeat_times}"
        )

    def output(self) -> TargetOnKart:
        return self.make_target(
            f"denserr/analyzer/damaged_analyzer/{self.log_name()}.pkl"
        )

    @classmethod
    def damage_doc(cls, original_text: str) -> Tuple[str, str]:
        sentences = nltk.sent_tokenize(original_text)

        if len(sentences) <= 1:
            return None
        perturbation = sentences.pop(random.randint(0, len(sentences) - 1))
        damaged = " ".join(sentences)

        return damaged, perturbation

    def repeat_n_return_max_score(
        self,
        query: str,
        ranking: list,
        retriever: DenseRetriever,
        corpus: dict,
        start_from: int = 100,
        until: int = 300,
    ) -> List[Tuple[int, int, float, str, str]]:
        p_score_perturbations = []
        for i, (docid, score) in enumerate(tqdm(ranking[start_from:until])):
            doc_text = corpus[docid]["text"]
            orig_rank = i + start_from
            damaged_result = self.damage_doc(doc_text)
            if damaged_result is None:
                continue
            damaged, perturbation = damaged_result
            perturbed_score = retriever.single_doc_score(query, damaged)
            new_rank = self.estimate_rank_pos(perturbed_score, ranking)
            p_score_perturbations.append(
                (new_rank, orig_rank, perturbed_score, damaged, perturbation)
            )

        p_score_perturbations = sorted(p_score_perturbations, key=lambda x: x[0])
        return p_score_perturbations

    def batch_repeat_n_return_max_score(
        self,
        query: str,
        ranking: list,
        retriever: DenseRetriever,
        corpus: dict,
        start_from: int = 100,
        until: int = 300,
    ) -> List[Tuple[int, int, float, str, str]]:
        p_score_perturbations = []
        queries = [query] * (until - start_from)
        damaged_texts = []
        perturbations = []
        docids = []
        for i, (docid, score) in enumerate(ranking[start_from:until]):
            doc_text = corpus[docid]["text"]
            orig_rank = i + start_from
            damaged_result = self.damage_doc(doc_text)
            if damaged_result is None:
                continue
            damaged, perturbation = damaged_result
            if len(damaged) <= 0:
                print("perturbation:", perturbation)
                print("damaged:", damaged, f"({len(damaged)})")
            damaged_texts.append(damaged)
            perturbations.append(perturbation)
            docids.append(docid)

        if len(damaged_texts) <= 0:
            return []
        perturbed_scores = retriever.batch_single_doc_score(queries, damaged_texts)

        for i, (damaged, perturbation, perturbed_score, docid) in enumerate(
            zip(damaged_texts, perturbations, perturbed_scores, docids)
        ):
            orig_rank = i + start_from
            new_rank = self.estimate_rank_pos(perturbed_score, ranking)
            p_score_perturbations.append(
                (new_rank, orig_rank, perturbed_score, damaged, perturbation)
            )

        p_score_perturbations = sorted(p_score_perturbations, key=lambda x: x[0])
        return p_score_perturbations

    def requires(self) -> Tuple[GokartTask, GokartTask]:
        if self.with_pyserini:
            ranking_task = Rerank()
        else:
            ranking_task = Retrieve()
        return ranking_task, LoadDataset()

    def run(self) -> None:
        print(f"cache output path: {self.output()._path()}")
        results, (corpus_loader, queries, _) = self.load()

        retriever = LoadRetriever(self.dataset_name, self.model_name).load_retriever()

        qids = list(results.keys())
        logger.debug(f"qids: {qids}")
        logger.debug(f"len qids: {len(qids)}")

        corpus = corpus_loader.to_dict()

        damaged_result = {}

        cache_path = cache_dir.joinpath("damaged_analyze", self.log_name())
        iter_cacher = IterCacher(cache_path)

        for i, (cacher, qid, p_score_perturbations) in enumerate(
            iter_cacher.iter(tqdm(qids, desc="iter qids"))
        ):
            if p_score_perturbations is not None:
                damaged_result[qid] = p_score_perturbations
                continue

            # print("*******************************")
            # print(qid)
            # print("*******************************")
            query = queries[qid]
            logger.debug(f"query: {query} ({qid})")

            # corpus = corpus_loader.fetch_docs(set(results[qid].keys()))

            ranking = sorted(results[qid].items(), key=lambda x: x[1], reverse=True)
            print(f"ranking len: {len(ranking)}")

            at_k = max(0, self.target_doc_rank - 1)
            target_doc_id, original_score = ranking[at_k]
            # logger.debug(
            #     f"target_doc length: {len(nltk.word_tokenize(target_doc['text']))}"
            # )
            repeat_runner = (
                self.batch_repeat_n_return_max_score
                if hasattr(retriever, "batch_single_doc_score")
                else self.repeat_n_return_max_score
            )

            p_score_perturbations = repeat_runner(
                query,
                ranking,
                retriever,
                corpus,
                start_from=self.damaged_start_at,
                until=self.damaged_until,
            )
            cacher.cache(p_score_perturbations)
            damaged_result[qid] = p_score_perturbations

        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        logger.info(f"timestamp: {now}")

        f"{self.sample_repeat_times}.pkl"
        writeout_json_to_log(
            damaged_result,
            "analyze_result",
            dir_name=f"{self.dataset_name}/{self.log_name()}_{now}",
        )

        self.dump(damaged_result)


def yield_analyze_result(
    analyze_result: Dict[str, list]
) -> Generator[tuple, None, None]:
    qids = analyze_result["qids"]
    new_ranks = analyze_result["new_ranks"]
    orig_ranks = analyze_result["orig_ranks"]
    perturbed_list = analyze_result["perturbed"]
    perturbations = analyze_result["perturbations"]
    perturbed_scores = analyze_result["perturbed_scores"]
    for qid, n_rank, orig_rank, perturbed, perturbation, perturbed_score in sorted(
        zip(
            qids, new_ranks, orig_ranks, perturbed_list, perturbations, perturbed_scores
        ),
        key=lambda x: x[1],
        reverse=False,
    ):
        yield qid, n_rank, orig_rank, perturbed, perturbation, perturbed_score


@inherits_config_params(DenseErrConfig)
class AnalyzeDamagedDistribution(GokartTask):
    dataset_name = luigi.Parameter()
    model_name = luigi.Parameter()
    with_pyserini = luigi.BoolParameter()

    sample_repeat_times = luigi.IntParameter(100)
    target_doc_rank = luigi.IntParameter(1)
    damaged_start_at = luigi.IntParameter()
    damaged_until = luigi.IntParameter()

    def requires(self) -> Tuple[GokartTask, GokartTask]:
        return DamagedAnalyze(rerun=self.rerun), Retrieve()

    def fig_save_dir(self) -> Path:
        return Path(__file__).parent.parent.parent.joinpath("tmp/figs/")

    def plot_basic_rank_distr(self, ranking_shifts: List[int]) -> None:
        fig = plt.figure(figsize=(24, 7))
        ax = fig.add_subplot(1, 1, 1)

        ax.hist(ranking_shifts, bins=10000)
        ax.set_title(
            f"{self.model_name} on {self.dataset_name} Standardized ranking shift distribution"
        )

        save_fig_path = self.fig_save_dir().joinpath(
            f"DamagedDistribution/{self.dataset_name}_{self.model_name}.png"
        )
        fig.savefig(save_fig_path)

    def plot_scores_distr(
        self,
        visualizer: DistributionVisualizer,
        scores: List[List[float]],
        labels: List[str],
        save_dirname: str = "ScoreDistr",
    ) -> None:
        title = "\n".join(
            [f"{self.model_name} on {self.dataset_name} score distribution"]
        )

        save_fig_path = self.fig_save_dir().joinpath(
            f"{save_dirname}/"
            f"{self.dataset_name}/{self.model_name}_@{self.target_doc_rank}_"
            f"{self.damaged_start_at}-{self.damaged_until}_{self.sample_repeat_times}"
            f"{'normalized'}"
            f".png"
        )
        visualizer.plot_score_distr(
            score_lists=scores,
            save_fig_path=save_fig_path,
            labels=labels,
            title=title,
            normalize=False,
        )

    def plot_shift_frequency_distr(
        self,
        visualizer: DistributionVisualizer,
        ranking_shifts: Union[List[int], List[float]],
        hist_min: Optional[Union[int, float]] = None,
        hist_max: Optional[Union[int, float]] = None,
        step: Union[int, float] = 25,
        density: bool = True,
        save_dirname: str = "RankshftFreqDistr",
        y_lim: Optional[float] = None,
    ) -> None:
        if hist_min is None:
            hist_min = -self.damaged_until
        if hist_max is None:
            hist_max = self.damaged_until
        title = "\n".join(
            [
                f"{self.model_name} on {self.dataset_name} rank shift frequency distribution",
                f"rank range: -10000 < -{hist_min}, x < {hist_max}, step: {step}",
                f"max rank shift: {max(ranking_shifts)}, total examples: {len(ranking_shifts)}",
            ]
        )

        save_fig_path = self.fig_save_dir().joinpath(
            f"{save_dirname}/"
            f"{self.dataset_name}/{self.model_name}_@{self.target_doc_rank}_"
            f"{self.damaged_start_at}-{self.damaged_until}_{self.sample_repeat_times}_"
            f"{hist_min}_{hist_max}_{step}_"
            f"{'dense' if density else ''}.png"
        )
        visualizer.plot_shift_frequency_distr(
            ranking_shifts_list=[ranking_shifts],
            save_fig_path=save_fig_path,
            labels=[self.model_name],
            hist_min=hist_min,
            hist_max=hist_max,
            step=step,
            density=density,
            title=title,
            y_lim=y_lim,
        )

    def run(self) -> None:
        damaged_result, results = self.load()
        visualizer = DistributionVisualizer(self.damaged_start_at, self.damaged_until)
        (
            ranking_shifts,
            normalized_ranking_shifts,
            new_ranks,
            orig_ranks,
        ) = visualizer.calc_shift_ranks(damaged_result)
        orig_norm_scores = visualizer.normalize_score_distr(
            results, from_retrieval_result=True
        )
        damaged_norm_scores = visualizer.normalize_score_distr(damaged_result)

        self.plot_basic_rank_distr(ranking_shifts)
        self.plot_shift_frequency_distr(visualizer, ranking_shifts)
        self.plot_shift_frequency_distr(
            visualizer,
            normalized_ranking_shifts,
            hist_min=0,
            hist_max=1,
            step=0.025,
            density=True,
            save_dirname="NRSDistr",
        )
        self.plot_scores_distr(
            visualizer, [orig_norm_scores, damaged_norm_scores], ["original", "damaged"]
        )

        for ranking_shift, orig_rank, new_rank in sorted(
            zip(ranking_shifts, orig_ranks, new_ranks), key=lambda x: x[0], reverse=True
        )[:10]:
            print(ranking_shift, ":", orig_rank, "=>", new_rank)


@inherits_config_params(DenseErrConfig)
class ShowDamagedCases(GokartTask):
    dataset_name = luigi.Parameter()
    model_name = luigi.Parameter()
    with_pyserini = luigi.BoolParameter()

    sample_repeat_times = luigi.IntParameter(100)
    target_doc_rank = luigi.IntParameter(1)
    damaged_start_at = luigi.IntParameter()
    damaged_until = luigi.IntParameter()

    def requires(self) -> Tuple[GokartTask, GokartTask, GokartTask]:
        return DamagedAnalyze(), Retrieve(), LoadDataset()

    def run(self):
        damaged_result, results, (_, queries, _) = self.load()
        for qid in tqdm(damaged_result):
            for (
                new_rank,
                orig_rank,
                perturbed_score,
                damaged,
                perturbation,
            ) in damaged_result[qid]:
                rank_shift = orig_rank - new_rank
                if rank_shift >= 200:
                    print("|||")
                    print("|-|-|")
                    print("|クエリ    |", queries[qid], "|")
                    print("|順位変化  |", f"{orig_rank} → {new_rank}: {rank_shift}", "|")
                    print("|削除後     |", damaged.replace("\n", "\\n"), "|")
                    print("|削除した文|", perturbation.replace("\n", "\\n"), "|")
                    print("--------------------")


@inherits_config_params(DenseErrConfig)
class SynthesizeDamagedRetrievalResult(GokartTask):
    dataset_name = luigi.Parameter()
    model_name = luigi.Parameter()
    with_pyserini = luigi.BoolParameter()

    sample_repeat_times = luigi.IntParameter(100)
    target_doc_rank = luigi.IntParameter(1)
    damaged_start_at = luigi.IntParameter()
    damaged_until = luigi.IntParameter()

    def requires(self) -> Tuple[GokartTask, GokartTask]:
        return DamagedAnalyze(), Retrieve()

    def run(self):
        damaged_result, results = self.load()
        damaged_retrieval_results = {}
        for qid in tqdm(damaged_result):
            ranking = sorted(results[qid].items(), key=lambda x: x[1], reverse=True)
            damaged_retrieval_results[qid] = results[qid].copy()
            for (
                new_rank,
                orig_rank,
                perturbed_score,
                damaged,
                perturbation,
            ) in damaged_result[qid]:
                docid, orig_score = ranking[orig_rank]
                damaged_retrieval_results[qid][docid] = perturbed_score
        return damaged_retrieval_results


from beir.retrieval.evaluation import EvaluateRetrieval


@inherits_config_params(DenseErrConfig)
class DamagedEvaluate(GokartTask):
    dataset_name = luigi.Parameter()
    model_name = luigi.Parameter()

    use_pyterrier = luigi.BoolParameter()

    def requires(self) -> Tuple[GokartTask, GokartTask]:
        return LoadDataset(), SynthesizeDamagedRetrievalResult()

    def output(self) -> TargetOnKart:
        return self.make_target(
            f"denserr/evaluate/{self.dataset_name}/{self.model_name}.pkl"
        )

    def run(self) -> None:
        print(f"cache output path: {self.output()._path()}")
        (_, _, qrels), results = self.load()
        for qid in results:
            docids = list(results[qid].keys())
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
