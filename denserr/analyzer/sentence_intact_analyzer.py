import datetime
import logging
import math
import random
from logging import getLogger
from typing import Generator, Optional, Union, List, Tuple, Dict
from pathlib import Path
from collections import defaultdict
import json

import luigi
import numpy as np
import nltk
from gokart.config_params import inherits_config_params
from tqdm import tqdm
import matplotlib.pyplot as plt
from gokart.target import TargetOnKart
from sklearn.preprocessing import StandardScaler

from denserr.config import DenseErrConfig
from denserr.dataset.load_dataset import LoadDataset
from denserr.model.load_model import DenseRetriever, LoadRetriever
from denserr.retrieve import Rerank, Retrieve
from denserr.utils.template import GokartTask
from denserr.utils.util import (
    writeout_json_to_log,
    now_log_dir,
    project_dir,
    jsons_dir,
    IterCacher,
    cache_dir,
)
from denserr.analyzer.distribution_visualizer import DistributionVisualizer

logger = getLogger(__name__)


@inherits_config_params(DenseErrConfig)
class SentenceInstactAnalyze(GokartTask):
    dataset_name = luigi.Parameter()
    model_name = luigi.Parameter()
    with_pyserini = luigi.BoolParameter()

    sample_repeat_times = luigi.IntParameter(100)
    target_doc_rank = luigi.IntParameter(1)

    intact_start_at = luigi.IntParameter()
    intact_until = luigi.IntParameter()

    perturb_context = luigi.Parameter()
    perturb_position = luigi.Parameter("random")

    def estimate_rank_pos(
        self, target_score: float, ranking: List[Tuple[str, float]]
    ) -> int:
        for i, (_, i_score) in enumerate(ranking):
            if target_score > i_score:
                return i + 1
        return len(ranking)

    def log_name(self) -> str:
        name = (
            f"{self.dataset_name}_{self.model_name}_@{self.target_doc_rank}_"
            f"{self.intact_start_at}-{self.intact_until}_{self.sample_repeat_times}_{self.perturb_context}"
        )
        if self.perturb_position == "end":
            return name
        elif self.perturb_position == "random":
            return f"{name}_{self.perturb_position}"
        else:
            raise ValueError(f"Unknown perturb position: {self.perturb_position}")

    def output(self) -> TargetOnKart:
        return self.make_target(f"denserr/analyzer/sent_intact/{self.log_name()}.pkl")

    def perturb(self, target_doc: str, corpus: dict, ranking: list) -> Tuple[str, str]:
        perturbation = self.sample_perturbation(corpus, ranking)
        if perturbation is None:
            return None

        if self.perturb_position == "end":
            perturbed = target_doc + perturbation
        elif self.perturb_position == "random":
            sentences = nltk.sent_tokenize(target_doc)
            insert_pos = random.randint(0, len(sentences) - 1)
            sentences.insert(insert_pos, perturbation)
            perturbed = " ".join(sentences)
        return perturbed, perturbation

    def sample_perturbation(self, corpus: dict, ranking: list) -> Optional[str]:
        if self.perturb_context == "ranking":
            seed_docid = random.choice([docid for docid, socre in ranking])
        elif self.perturb_context == "corpus":
            seed_docid = random.choice(list(corpus.keys()))

        seed_doc_text = corpus[seed_docid]["text"]
        sentences = nltk.sent_tokenize(seed_doc_text)
        if len(sentences) <= 0:
            return None

        perturbation = random.choice(sentences)
        return perturbation

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
            perturbed_result = self.perturb(doc_text, corpus, ranking)
            if perturbed_result is None:
                continue

            perturbed, perturbation = perturbed_result
            if perturbed == perturbation:
                continue
            perturbed_score = retriever.single_doc_score(query, perturbed)
            new_rank = self.estimate_rank_pos(perturbed_score, ranking)
            p_score_perturbations.append(
                (new_rank, orig_rank, perturbed_score, perturbed, perturbation)
            )
            if new_rank < 100:
                print(f"new_rank: {orig_rank} -> {new_rank}")

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
        perturbed_texts = []
        perturbations = []
        docids = []
        for i, (docid, score) in enumerate(ranking[start_from:until]):
            doc_text = corpus[docid]["text"]
            orig_rank = i + start_from
            perturbed_result = self.perturb(doc_text, corpus, ranking)
            if perturbed_result is None:
                continue

            perturbed, perturbation = perturbed_result
            if perturbed == perturbation:
                continue
            perturbed_texts.append(perturbed)
            perturbations.append(perturbation)
            docids.append(docid)

        if len(perturbed_texts) <= 0:
            return []
        perturbed_scores = retriever.batch_single_doc_score(queries, perturbed_texts)

        for i, (perturbed, perturbation, perturbed_score, docid) in enumerate(
            zip(perturbed_texts, perturbations, perturbed_scores, docids)
        ):
            orig_rank = i + start_from
            new_rank = self.estimate_rank_pos(perturbed_score, ranking)
            p_score_perturbations.append(
                (new_rank, orig_rank, perturbed_score, perturbed, perturbation)
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
        results, (corpus_loader, queries, _) = self.load()

        retriever = LoadRetriever(self.dataset_name, self.model_name).load_retriever()

        qids = list(results.keys())
        logger.debug(f"qids: {qids}")
        logger.debug(f"len qids: {len(qids)}")

        corpus = corpus_loader.to_dict()

        perturbed_result = {}
        cache_path = cache_dir.joinpath("sentence_intact_analysis", self.log_name())
        iter_cacher = IterCacher(cache_path)

        for i, (cacher, qid, p_score_perturbations) in enumerate(
            iter_cacher.iter(tqdm(qids, desc="iter qids"))
        ):
            print("*******************************")
            print(qid)
            print("*******************************")
            if p_score_perturbations is not None:
                perturbed_result[qid] = p_score_perturbations
                continue

            query = queries[qid]
            logger.debug(f"query: {query} ({qid})")

            ranking = sorted(results[qid].items(), key=lambda x: x[1], reverse=True)
            print(f"ranking len: {len(ranking)}")

            # corpus = corpus_loader.fetch_docs(set(results[qid].keys()))

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
                start_from=self.intact_start_at,
                until=self.intact_until,
            )
            cacher.cache(p_score_perturbations)
            perturbed_result[qid] = p_score_perturbations

        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        logger.info(f"timestamp: {now}")

        f"{self.sample_repeat_times}.pkl"
        writeout_json_to_log(
            perturbed_result,
            "analyze_result",
            dir_name=f"{self.dataset_name}/{self.log_name()}_{now}",
        )

        self.dump(perturbed_result)


@inherits_config_params(DenseErrConfig)
class AnalyzeSentenceInstactDistribution(GokartTask):
    dataset_name = luigi.Parameter()
    model_name = luigi.Parameter()
    with_pyserini = luigi.BoolParameter()

    sample_repeat_times = luigi.IntParameter(100)
    target_doc_rank = luigi.IntParameter(1)

    intact_start_at = luigi.IntParameter()
    intact_until = luigi.IntParameter()

    perturb_context = luigi.Parameter()

    def requires(self) -> Tuple[GokartTask, GokartTask]:
        return SentenceInstactAnalyze(), Retrieve()

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
            f"SentIntactDistribution/{self.dataset_name}_{self.model_name}.png"
        )
        if not save_fig_path.parent.exists():
            save_fig_path.parent.mkdir(parents=True)
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
            f"{self.intact_start_at}-{self.intact_until}_{self.sample_repeat_times}"
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
            hist_min = -self.intact_until
        if hist_max is None:
            hist_max = self.intact_until
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
            f"{self.intact_start_at}-{self.intact_until}_{self.sample_repeat_times}_"
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
        visualizer = DistributionVisualizer(self.intact_start_at, self.intact_until)
        (
            ranking_shifts,
            normalized_ranking_shifts,
            new_ranks,
            orig_ranks,
        ) = visualizer.calc_shift_ranks(damaged_result, is_intact=True)
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
            save_dirname="SentIntactNRSDistr",
        )
        self.plot_scores_distr(
            visualizer,
            [orig_norm_scores, damaged_norm_scores],
            ["original", "perturbed"],
        )

        for ranking_shift, orig_rank, new_rank in sorted(
            zip(ranking_shifts, orig_ranks, new_ranks), key=lambda x: x[0], reverse=True
        )[:10]:
            print(ranking_shift, ":", orig_rank, "=>", new_rank)


@inherits_config_params(DenseErrConfig)
class ShowSentIntactCases(GokartTask):
    dataset_name = luigi.Parameter()
    model_name = luigi.Parameter()
    with_pyserini = luigi.BoolParameter()

    sample_repeat_times = luigi.IntParameter(100)
    target_doc_rank = luigi.IntParameter(1)

    intact_start_at = luigi.IntParameter()
    intact_until = luigi.IntParameter()

    perturb_context = luigi.Parameter()

    def requires(self) -> Tuple[GokartTask, GokartTask, GokartTask]:
        return SentenceInstactAnalyze(), Retrieve(), LoadDataset()

    def run(self):
        damaged_result, results, (_, queries, _) = self.load()
        high_rankshift_results = {}
        for qid in tqdm(damaged_result):
            for result in damaged_result[qid]:
                (
                    new_rank,
                    orig_rank,
                    perturbed_score,
                    damaged,
                    perturbation,
                ) = result
                rank_shift = new_rank - orig_rank
                if rank_shift >= 200:
                    high_rankshift_results[qid] = [
                        new_rank,
                        orig_rank,
                        perturbed_score,
                        damaged,
                        perturbation,
                        rank_shift,
                    ]
        high_rankshift_results = sorted(
            high_rankshift_results.items(), key=lambda x: x[1][-1]
        )
        for qid, (
            new_rank,
            orig_rank,
            perturbed_score,
            perturbed,
            perturbation,
            rank_shift,
        ) in high_rankshift_results:
            print("|||")
            print("|-|-|")
            print("|クエリ    |", queries[qid], "|")
            print("|順位変化  |", f"{orig_rank} → {new_rank}: {rank_shift}", "|")
            print("|追加後     |", perturbed.replace("\n", "\\n"), "|")
            print("|追加した文|", perturbation.replace("\n", "\\n"), "|")
            print("--------------------")


@inherits_config_params(DenseErrConfig)
class CalcSentIntactStats(GokartTask):
    dataset_name = luigi.Parameter()
    model_name = luigi.Parameter()
    with_pyserini = luigi.BoolParameter()

    sample_repeat_times = luigi.IntParameter(100)
    target_doc_rank = luigi.IntParameter(1)

    intact_start_at = luigi.IntParameter()
    intact_until = luigi.IntParameter()

    perturb_context = luigi.Parameter()

    def log_name(self) -> str:
        return (
            f"{self.dataset_name}_{self.model_name}_@{self.target_doc_rank}_"
            f"{self.intact_start_at}-{self.intact_until}_{self.sample_repeat_times}_{self.perturb_context}"
        )

    def json_path(self) -> Path:
        return jsons_dir.joinpath(f"{self.log_name()}.json")

    def requires(self) -> Tuple[GokartTask, GokartTask, GokartTask]:
        return SentenceInstactAnalyze(), Retrieve(), LoadDataset()

    def run(self):
        intact_result, results, (_, queries, _) = self.load()

        stats = defaultdict(list)
        for qid in tqdm(intact_result):
            for (
                new_rank,
                orig_rank,
                perturbed_score,
                perturbed,
                perturbation,
            ) in intact_result[qid]:
                perturbation_len = len(nltk.word_tokenize(perturbation))
                perturbed_len = len(nltk.word_tokenize(perturbed))
                original_text_len = perturbed_len - perturbation_len

                perturbation_percentage = perturbation_len / original_text_len

                stats["perturbation_len"].append(perturbation_len)
                stats["perturbed_len"].append(perturbed_len)
                stats["original_text_len"].append(original_text_len)
                stats["perturbation_percentage"].append(perturbation_percentage)

        ave_stats = {}
        for key in stats:
            ave_stats[f"ave_{key}"] = sum(stats[key]) / len(stats[key])

        stats.update(ave_stats)
        logger.debug(ave_stats)

        self.json_path().write_text(json.dumps(stats, ensure_ascii=False, indent=2))

        self.dump(stats)
