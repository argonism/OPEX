import datetime
import logging
import math
import random
from logging import getLogger
from typing import Generator, Optional, Union, List, Dict, Tuple, Any
from pathlib import Path

import luigi
import numpy as np
import nltk
from gokart.config_params import inherits_config_params
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns

import matplotlib as mpl

mpl.rcParams["pdf.fonttype"] = 42
logger = logging.getLogger("fontTools.subset")
logger.setLevel(logging.ERROR)

logger = getLogger(__name__)


class DistributionVisualizer:
    def __init__(self, damaged_start_at: int, damaged_until: int) -> None:
        self.damaged_start_at = damaged_start_at
        self.damaged_until = damaged_until

    def normalize_score_distr(
        self,
        result: dict,
        from_retrieval_result: bool = False,
    ) -> List[float]:
        def yield_scores_from_damaged(
            damaged_qid_result: List,
        ) -> Generator[float, None, None]:
            for (
                _,
                _,
                perturbed_score,
                _,
                _,
            ) in damaged_qid_result:
                yield perturbed_score

        normalized_score_distr = []
        for qid in tqdm(result):
            scores: List[float] = []
            if from_retrieval_result:
                score_iter = result[qid].values()
            else:
                score_iter = yield_scores_from_damaged(result[qid])
            for score in score_iter:
                scores.append(score)
            if len(scores) <= 0:
                continue
            scores = np.array(scores).reshape(-1, 1)
            scores = StandardScaler().fit_transform(scores).squeeze()
            if len(scores.shape) <= 0:
                scores = np.array([scores])
            # print(scores)
            # print(scores.shape)
            # print(len(scores.shape))
            normalized_score_distr = np.concatenate([normalized_score_distr, scores], 0)
        return normalized_score_distr

    def calc_shift_ranks(
        self, damaged_result: List, is_intact: bool = False
    ) -> Tuple[List[int], List[int], List[int], List[int]]:
        ranking_shifts = []
        normalized_ranking_shifts = []
        new_ranks = []
        orig_ranks = []
        for qid in tqdm(damaged_result):
            for (
                new_rank,
                orig_rank,
                perturbed_score,
                damaged,
                perturbation,
                *position,
            ) in damaged_result[qid]:
                if orig_rank - new_rank < -100000:
                    print(f"{qid} at {orig_rank}")
                    continue
                ranking_shift = (
                    new_rank - orig_rank if is_intact else orig_rank - new_rank
                )
                normalized_ranking_shift = (
                    ranking_shift / (orig_rank + 1)
                    if is_intact
                    else ranking_shift / (orig_rank - 1)
                )
                ranking_shifts.append(ranking_shift)
                normalized_ranking_shifts.append(normalized_ranking_shift)
                new_ranks.append(new_rank)
                orig_ranks.append(orig_rank)
        return ranking_shifts, normalized_ranking_shifts, new_ranks, orig_ranks

    def histgramize(
        self,
        array: Union[List[int], List[float]],
        hist_min: Optional[Union[int, float]] = None,
        hist_max: Optional[Union[int, float]] = None,
        step: Union[int, float] = 25,
        density: bool = True,
        include_under: bool = True,
        accumulate: bool = False,
        start_from: Optional[float] = None,
        for_paper: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if hist_min is None:
            hist_min = -self.damaged_until
        if hist_max is None:
            hist_max = self.damaged_until

        padding = 1 if isinstance(step, int) else 0.0001

        bin_edges_min = [-10000]
        bin_edges = np.arange(hist_min, hist_max + padding, step)
        bin_edges = np.concatenate([bin_edges_min, bin_edges])
        # if include_under:
        # else:
        #     array = [e for e in array if e > 0]
        #     bin_edges = np.arange(0, hist_max + padding, step)

        freq_distr, _ = np.histogram(array, bins=bin_edges)
        if density:
            freq_distr = freq_distr / sum(freq_distr)

        if accumulate:
            freq_distr = np.cumsum(freq_distr)

        if start_from is not None or not include_under:
            new_freq_distr, new_bin_edges = [], [bin_edges[0]]
            for e, edge in zip(freq_distr, bin_edges[1:]):
                threshold = 0 if start_from is None else start_from
                if edge > threshold:
                    new_freq_distr.append(e)
                    new_bin_edges.append(edge)
            freq_distr = new_freq_distr
            bin_edges = new_bin_edges

        freq_distr = [100 * (1 - freq) for freq in freq_distr]
        # if for_paper:
        #     freq_distr = [100 * (1 - freq) for freq in freq_distr]
        return freq_distr, bin_edges

    def plot_score_distr(
        self,
        score_lists: List[List[float]],
        save_fig_path: Union[str, Path],
        labels: Optional[List[str]] = None,
        normalize: bool = False,
        title: str = "",
    ) -> None:
        fig = plt.figure(figsize=(24, 7))
        ax = fig.add_subplot(1, 1, 1)

        if (labels is not None) and not len(labels) == len(score_lists):
            raise Exception(
                "label and scores lenth does not match:"
                + f" {len(labels)} and {len(score_lists)}",
            )
        labels_iter = [""] * len(score_lists) if labels is None else labels

        freq_distrs = []
        for i, (scores, label) in enumerate(zip(score_lists, labels_iter)):
            if normalize:
                scores = np.array(scores).reshape(1, -1)
                scores = StandardScaler().fit_transform(scores).squeeze()
            # print(scores.shape)
            # print(len([i + 1] * len(scores)))
            # ax.plot(scores, [i + 1] * len(scores), label=label)
            ax.hist(scores, label=label, bins=1000, alpha=0.4)

        # plt.xticks(rotation=-45)
        ax.set_title(title)
        if labels is not None:
            ax.legend()

        Path(save_fig_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_fig_path)
        logger.info(f"save fig at : {save_fig_path}")

    def plot_shift_frequency_distr(
        self,
        ranking_shifts_list: List[Union[List[int], List[float]]],
        save_fig_path: Union[str, Path],
        labels: Optional[List[str]] = None,
        hist_min: Optional[Union[int, float]] = None,
        hist_max: Optional[Union[int, float]] = None,
        step: Union[int, float] = 25,
        density: bool = True,
        title: str = "",
        y_lim: Optional[float] = None,
        include_under: bool = True,
        accumulate: bool = False,
        fig: Any = None,
        ax: Any = None,
        grayout_first: bool = True,
        for_paper: bool = False,
        xlabel: str = "Rank shift",
        ylabel: str = "Cumulative frequency",
    ) -> None:
        if fig is None:
            fig = plt.figure(figsize=(24, 7))
        if ax is None:
            ax = fig.add_subplot(1, 1, 1)

        if y_lim is not None:
            logger.info(f"setting y lim: {0} ~ {y_lim}")
            ax.set_ylim(0, y_lim)

        if (labels is not None) and not len(labels) == len(ranking_shifts_list):
            raise Exception(
                "label and ranking_shift_list lenth does not match:"
                + f" {len(labels)} and {len(ranking_shifts_list)}",
            )
        labels_iter = [""] * len(ranking_shifts_list) if labels is None else labels

        freq_distrs = []
        for i, (ranking_shifts, label) in enumerate(
            zip(ranking_shifts_list, labels_iter)
        ):
            freq_distr, bin_edges = self.histgramize(
                ranking_shifts,
                hist_min=hist_min,
                hist_max=hist_max,
                step=step,
                density=density,
                include_under=include_under,
            )
            if accumulate:
                freq_distr = np.cumsum(freq_distr)
            freq_distrs.append(freq_distr)
            # print(f"rank promoted propotion of {label}: {sum(freq_distr):.3f}")

            X_axis = np.arange(len(freq_distr))
            x_labels = [
                f"≤ {x:.3f}" if isinstance(x, float) else f"≤ {x}"
                for x in bin_edges[1:]
            ]
            width = (0.8 - 0.03) / len(labels_iter)
            offset = (width + 0.01) * i
            color = "0.75" if grayout_first and i <= 0 else None
            ax.bar(
                X_axis + offset,
                freq_distr,
                width=width,
                label=label,
                color=color,
            )

        # plt.xticks(rotation=-45)
        ax.set_title(title)
        # if not for_paper:
        #     ax.set_title(title)
        labelsize = 40 if for_paper else 25
        ax.set_xlabel(xlabel, fontsize=labelsize)
        ax.set_ylabel(xlabel, fontsize=labelsize)
        ax.tick_params(axis="x", labelsize=labelsize)
        ax.tick_params(axis="y", labelsize=labelsize - 5)
        if labels is not None:
            fontsize = 30 if for_paper else 30
            ax.legend(
                fontsize=fontsize,
            )

        Path(save_fig_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_fig_path, bbox_inches="tight", pad_inches=0)
        if for_paper:
            fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
            pdf_save_path = save_fig_path.parent.joinpath(f"{save_fig_path.stem}.pdf")
            fig.savefig(pdf_save_path, bbox_inches="tight", pad_inches=0)
        logger.info(f"save fig at : {save_fig_path}")

    def plot_acc_ranking_shift_line(
        self,
        ranking_shifts_list: List[Union[List[int], List[float]]],
        save_fig_path: Union[str, Path],
        labels: Optional[List[str]] = None,
        hist_min: Optional[Union[int, float]] = None,
        hist_max: Optional[Union[int, float]] = None,
        step: Union[int, float] = 25,
        density: bool = True,
        title: str = "",
        y_lim: float = 100,
        include_under: bool = True,
        accumulate: bool = True,
        start_from: Optional[int] = None,
        grayout_first: bool = True,
        fig: Any = None,
        ax: Any = None,
        for_paper: bool = False,
        xlabel: str = "$t$",
        ylabel: str = "${\\rm RS}_t$ (%)",
        line_styles: Optional[Dict] = None,
        color_pallet: str = "mako",
    ) -> None:
        logger.info(f"plot_acc_ranking_shift_line (ylim={y_lim})")
        sns.set_palette(color_pallet)
        if fig is None:
            fig = plt.figure(figsize=(12, 7))
        if ax is None:
            ax = fig.add_subplot(1, 1, 1)

        if y_lim is not None:
            logger.info(f"setting y_lim: ({0}, {y_lim})")
            ax.set_ylim(0, y_lim)

        if (labels is not None) and not len(labels) == len(ranking_shifts_list):
            raise Exception(
                "label and ranking_shift_list lenth does not match:"
                + f" {len(labels)} and {len(ranking_shifts_list)}",
            )
        labels_iter = [""] * len(ranking_shifts_list) if labels is None else labels
        # colors_iter = [None] * len(ranking_shifts_list) if line_styles is None else line_styles

        freq_distrs = []
        for i, (ranking_shifts, label) in enumerate(
            zip(ranking_shifts_list, labels_iter)
        ):
            freq_distr, bin_edges = self.histgramize(
                ranking_shifts,
                hist_min=hist_min,
                hist_max=hist_max,
                step=step,
                density=density,
                include_under=include_under,
                accumulate=accumulate,
                start_from=start_from,
                for_paper=for_paper,
            )
            freq_distrs.append(freq_distr)
            # print(f"rank promoted propotion of {label}: {sum(freq_distr):.3f}")

            # X_axis = np.arange(len(freq_distr))
            # x_labels = [
            #     f"≤ {x:.3f}" if isinstance(x, float) else f"≤ {x}"
            #     for x in bin_edges[1:]
            # ]

            line_style = (
                line_styles[label]
                if (line_styles is not None) and (label in line_styles)
                else {}
            )
            if "linewidth" not in line_style:
                line_style["linewidth"] = 4

            # if grayout_first and i <= 0:
            #     line_style["color"] = "0.25"
            ax.plot(bin_edges[1:], freq_distr, label=label, **line_style)
            # if start_from is not None:
            #     for edge, freq in zip(bin_edges[1:], freq_distr):
            #         if edge % 10 == 0:
            #             ax.text(edge, freq, f"{freq:.3f}")
            # ax.set_xticklabels(x_labels)

        # plt.xticks(rotation=-45)
        start, end, step = hist_min, hist_max, 50
        xtics = list(range(start, end + 1, step))
        ax.set_xticks(xtics)
        # ax.set_ylim(top=100)
        # ax.set_yticks([i for i in range(0, 21, 5)])
        ax.set_title(title, fontsize=40)
        ax.minorticks_on()

        labelsize = 35 if for_paper else 25
        ax.set_xlabel(xlabel, fontsize=labelsize)
        ax.set_ylabel(ylabel, fontsize=labelsize)
        ax.tick_params(axis="x", labelsize=labelsize)
        ax.tick_params(axis="y", labelsize=labelsize)
        if labels is not None:
            leg = ax.legend(
                fontsize=40 if for_paper else 30,
                ncol=2 if len(label) > 5 else 1,
                markerscale=2,
                handleheight=0.5,
                handletextpad=0.2,
                borderpad=0.2,
                columnspacing=0.2,
            )
            for legobj in leg.legendHandles:
                legobj.set_linewidth(6.0)

        Path(save_fig_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_fig_path, bbox_inches="tight", pad_inches=0)

        if for_paper:
            fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
            pdf_save_path = save_fig_path.parent.joinpath(f"{save_fig_path.stem}.pdf")
            fig.savefig(pdf_save_path, bbox_inches="tight", pad_inches=0)
        logger.info(f"save fig at : {save_fig_path}")
