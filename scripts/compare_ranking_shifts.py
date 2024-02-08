from typing import Generator, Optional, Union, Any, List, Dict
from pathlib import Path
from argparse import ArgumentParser, Namespace
import pickle
import sys
import logging
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.append(str(Path(__file__).parent.parent))
from denserr.analyzer.distribution_visualizer import DistributionVisualizer

MODEL_NAME_PATCH_TABLE = {
    "bm25-msmarco-doc": "BM25",
    "bm25-robust04": "BM25",
    "bm25": "BM25",
    "ptsplade": "SPLADE",
    "dpr": "DPR",
    "ance": "ANCE",
    "colbert": "ColBERT",
    "deepct": "DeepCT",
    "-sent-parallel": "-W1",
    "sent-": "",
    "-sent": "-W1",
    "-parallel": "",
    "-w": "-W",
}


class CompareRankshiftsDistribution:
    def __init__(
        self,
        result_pathes: List[Path],
        intact: bool = False,
        ignore_extra: bool = False,
        for_paper: bool = False,
    ) -> None:
        self.result_pathes = result_pathes
        self.is_intact = intact
        self.ignore_extra = ignore_extra
        self.for_paper = for_paper

        result_infos = self.validate_same_setting()
        self.base_info = result_infos[0]
        logger.info(self.base_info)
        self.result_infos = result_infos
        self.labels = [info["model_name"] for info in result_infos]
        self.labels = self.patch_model_name(self.labels)
        print(self.labels)

        self.dataset_name = self.base_info["dataset"]
        self.base_model_name = self.base_info["model_name"]
        self.damaged_start_at = self.base_info["start_at"]
        self.damaged_until = self.base_info["end_at"]
        self.target_rank = self.base_info["target_rank"]
        self.sample_repeat_times = self.base_info["sample_repeat_times"]
        self.extras = self.base_info["extras"]

    @classmethod
    def patch_model_name(cls, labels) -> List[str]:
        patched_labels = []
        for label in labels:
            label = label.lower()
            for model_name, patch in MODEL_NAME_PATCH_TABLE.items():
                if model_name in label:
                    label = label.replace(model_name, patch)
            patched_labels.append(label)
        return patched_labels

    def filepath_to_info(self, path: Path) -> Dict:
        path_stem = path.stem
        (
            dataset,
            model_name,
            target_rank,
            start_at_end_at,
            sample_repeat_times,
            etc,
        ) = path_stem.split("_", maxsplit=5)
        extras = []
        if self.is_intact:
            perturb_context, _ = etc.rsplit("_", maxsplit=1)
            extras += perturb_context.split("_")
        start_at, end_at = start_at_end_at.split("-")
        return {
            "dataset": dataset,
            "model_name": model_name,
            "target_rank": target_rank.replace("@", ""),
            "start_at": int(start_at),
            "end_at": int(end_at),
            "sample_repeat_times": int(sample_repeat_times),
            "extras": extras,
        }

    @property
    def model_names_text(self, split_threshold: int = 10) -> str:
        if len(self.labels) > split_threshold:
            names_texts = []
            for i in range(len(self.labels) // split_threshold):
                names_text = "&".join(
                    self.labels[(i) * split_threshold : (i + 1) * split_threshold]
                )
                names_texts.append(names_text)
            return "/".join(names_texts)
        else:
            return "&".join(self.labels)

    def line_styles(self, model_names: List[str]) -> Dict[str, Dict[str, str]]:
        if self.is_windowed:
            return None

        colors = sns.color_palette("tab10", n_colors=4)
        PALLET = {
            "BM25": {"color": "k", "linewidth": 3},
            "ANCE": {"color": colors[0]},
            "ColBERT": {"color": colors[1]},
            "DeepCT": {"color": colors[2]},
            "SPLADE": {"color": colors[3]},
            "-W": {"linestyle": (0, (5, 2))},
            # "ANCE-W1": {"color": "#edcdab"},
            # "ANCE-W5": {"linestyle": (0, (5, 2, 1, 2)), "color": "#dfa86d"},
            # "ANCE-W10": {"linestyle": (0, (5, 2)), "color": "#d1822e"},
            # "ANCE-W15": {"linestyle": (0, (5, 2, 1, 2)), "color": "#925b20"},
            # "ANCE-W20": {"linestyle": (0, (5, 2)), "color": "#543412"},
            # "SPLADE-W1": {"color": "#adebcc"},
            # "SPLADE-W5": {"linestyle": (0, (5, 2, 1, 2)), "color": "#70dba6"},
            # "SPLADE-W10": {"linestyle": (0, (5, 2)), "color": "#33cc80"},
            # "SPLADE-W15": {"linestyle": (0, (5, 2, 1, 2)), "color": "#248f59"},
            # "SPLADE-W20": {"linestyle": (0, (5, 2)), "color": "#145233"},
            # "-W5": {"linestyle": "--", "alpha": 0.20},
            # "-W10": {"linestyle": "--", "alpha": 0.35},
            # "-W15": {"linestyle": "--", "alpha": 0.55},
            # "-W20": {"linestyle": "--", "alpha": 0.75},
        }
        styles = {}
        for model_name in model_names:
            styles[model_name] = {}
            for name_pattern in PALLET:
                if name_pattern in model_name:
                    styles[model_name].update(**PALLET[name_pattern])
        return styles

    def gen_figs_title(
        self,
        fig_name: str,
        hist_min: Union[int, float],
        hist_max: Union[int, float],
        step: Union[int, float],
    ) -> str:
        if self.for_paper:
            analysis_method = "Addition" if self.is_intact else "Deletion"
            dataset = ""
            if self.dataset_name == "robust04":
                dataset = "Robust04"
            elif self.dataset_name == "dl19-doc":
                dataset = "TREC DL 19"
            elif self.dataset_name == "dl20-doc":
                dataset = "TREC DL 20"
            elif self.dataset_name == "msmarco-doc":
                dataset = "MS MARCO"
            else:
                raise Exception(f"Unknown dataset: {self.dataset_name}")
            # title = f"Sentence {analysis_method} Analysis on {dataset}"
            title = f"{dataset}"
            return title

        title = "\n".join(
            [
                f"intact-{'_'.join(self.extras)}" if self.is_intact else "",
                f"{self.model_names_text} on {self.dataset_name} {fig_name}",
                f"rank range: -10000 < -{hist_min}, x < {hist_max}, step: {step}",
            ]
        )
        return title

    @property
    def is_sent(self):
        return any(["W1" in label for label in self.labels])

    @property
    def is_windowed(self):
        return any(["W5" in label for label in self.labels])

    def gen_filename(
        self,
        hist_min: Union[int, float],
        hist_max: Union[int, float],
        step: Union[int, float],
        density: bool,
        additional: Optional[str] = None,
    ) -> str:
        if self.for_paper:
            analyze_method = "Intact" if self.is_intact else "Damaged"
            is_sent = "_Sent" if self.is_sent else ""
            windowed_label = "_Window" if self.is_windowed else ""
            model_family = self.labels[0]
            save_fig_filename = (
                f"for_paper/{analyze_method}_{self.dataset_name}"
                f"{is_sent}{windowed_label}"
                f"_{model_family if self.is_windowed else ''}"
                f"{hist_min}_{hist_max}_{step}"
                f"{'_' + additional if additional is not None else ''}"
                f"{'_'.join(self.extras)}.png"
            )
            return save_fig_filename
        save_fig_filename = (
            f"{self.dataset_name}/{self.model_names_text}_@{self.target_rank}_"
            f"{self.damaged_start_at}-{self.damaged_until}_{self.sample_repeat_times}_"
            f"{hist_min}_{hist_max}_{step}"
            f"{'_dense' if density else ''}"
            f"{'_' + additional if additional is not None else ''}"
            f"{'_' + '_'.join(self.extras) if self.is_intact else ''}"
            ".png"
        )
        return save_fig_filename

    def accum_freq_distr_table(
        self,
        visualizer: DistributionVisualizer,
        ranking_shifts_list: List[Union[List[int], List[float]]],
        labels: List[str],
        hist_min: Union[int, float] = 0,
        hist_max: Union[int, float] = 1,
        step: Union[int, float] = 0.025,
        density: bool = True,
        include_under: bool = True,
        save_dir: Path = Path(__file__).parent.joinpath("damaged_ranking_shifts_freq"),
    ) -> None:
        fig = plt.figure(figsize=(24, 7))
        ax = fig.add_subplot(1, 1, 1)

        table = defaultdict(lambda: defaultdict(float))
        for i, (ranking_shifts, label) in enumerate(zip(ranking_shifts_list, labels)):
            freq_distr, bin_edges = visualizer.histgramize(
                ranking_shifts,
                hist_min=hist_min,
                hist_max=hist_max,
                step=step,
                density=density,
                include_under=include_under,
            )
            for i, freq in enumerate(freq_distr):
                le = "$\le$" if self.for_paper else "≤"
                freq_range = f"{le} {bin_edges[i+1]}"
                table[freq_range][label] = 100 - freq

        header_line = "|NRS|"
        spacer_line = "| --------------- |"
        for lable in labels:
            header_line += f"{lable}|"
            spacer_line += " --- |"
        print(header_line)
        print(spacer_line)
        for freq in table:
            column = f"|{freq}|"
            for label in table[freq]:
                column += f"$^\\ast${table[freq][label]:.3f}|"
            print(column)

        fig_title = "accumlated rank shift frequency distribution"
        title = self.gen_figs_title(fig_title, hist_min, hist_max, step)

        save_dir.mkdir(parents=True, exist_ok=True)

        save_fig_filename = self.gen_filename(hist_min, hist_max, step, density)
        save_fig_path = save_dir.joinpath(save_fig_filename)

        plt.xticks(rotation=-45)
        ax.set_title(title)
        ax.minorticks_on()
        ax.grid(which="major", color="0.8")
        ax.grid(which="minor", color="0.9", linestyle="--")
        ax.tick_params(axis="x", labelsize=14)
        ax.tick_params(axis="y", labelsize=14)
        if labels is not None:
            ax.legend(fontsize=20)

        Path(save_fig_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_fig_path)
        logger.info(f"save fig at : {save_fig_path}")

    def validate_same_setting(self) -> tuple:
        logger.info("validating both cached results are under the same setting ...")
        base_info = self.filepath_to_info(self.result_pathes[0])
        result_infos = [base_info]
        for target_file in self.result_pathes[1:]:
            target_info = self.filepath_to_info(target_file)
            for info_key in base_info:
                if info_key in ["model_name"]:
                    continue
                if info_key == "extras" and self.ignore_extra:
                    continue
                logger.debug(f"validating: {info_key}")
                if not base_info[info_key] == target_info[info_key]:
                    print("base_info:", base_info)
                    print("target_info:", target_info)
                    raise Exception(
                        f"setting does not match at {target_file}! {base_info[info_key]} !== {target_info[info_key]}"
                    )
            result_infos.append(target_info)
        logger.info("all setting are matched")
        return result_infos

    def plot_shift_frequency_distr(
        self,
        visualizer: DistributionVisualizer,
        ranking_shifts: List[Union[List[int], List[float]]],
        hist_min: Optional[Union[int, float]] = None,
        hist_max: Optional[Union[int, float]] = None,
        step: Union[int, float] = 25,
        density: bool = True,
        save_dir: Path = Path(__file__).parent.joinpath("damaged_ranking_shifts"),
        y_lim: Optional[float] = None,
        include_under: bool = True,
    ) -> None:
        if hist_min is None:
            hist_min = -self.damaged_until
        if hist_max is None:
            hist_max = self.damaged_until

        fig_title = "rank shift frequency distribution"
        title = self.gen_figs_title(fig_title, hist_min, hist_max, step)

        save_dir.mkdir(parents=True, exist_ok=True)

        save_fig_filename = self.gen_filename(hist_min, hist_max, step, density)
        save_fig_path = save_dir.joinpath(save_fig_filename)
        visualizer.plot_shift_frequency_distr(
            ranking_shifts_list=ranking_shifts,
            save_fig_path=save_fig_path,
            labels=self.labels,
            hist_min=hist_min,
            hist_max=hist_max,
            step=step,
            density=density,
            title=title,
            y_lim=y_lim,
            include_under=include_under,
        )

    def plot_acc_shift_frequency_distr_line(
        self,
        visualizer: DistributionVisualizer,
        ranking_shifts: List[Union[List[int], List[float]]],
        hist_min: Optional[Union[int, float]] = None,
        hist_max: Optional[Union[int, float]] = None,
        step: Union[int, float] = 25,
        density: bool = True,
        save_dir: Path = Path(__file__).parent.joinpath(
            "damaged_ranking_shifts_acc_line"
        ),
        y_lim: Optional[float] = None,
        include_under: bool = True,
        start_from: Optional[int] = None,
        line_styles: List[str] = None,
        color_pallet: List[str] = None,
    ) -> None:
        if hist_min is None:
            hist_min = -self.damaged_until
        if hist_max is None:
            hist_max = self.damaged_until

        fig_title = "accumlated rank shift frequency distribution"
        title = self.gen_figs_title(fig_title, hist_min, hist_max, step)

        save_dir.mkdir(parents=True, exist_ok=True)

        additional = None if start_from is None else str(start_from)
        save_fig_filename = self.gen_filename(
            hist_min, hist_max, step, density, additional=additional
        )
        save_fig_path = save_dir.joinpath(save_fig_filename)
        visualizer.plot_acc_ranking_shift_line(
            ranking_shifts_list=ranking_shifts,
            save_fig_path=save_fig_path,
            labels=self.labels,
            hist_min=hist_min,
            hist_max=hist_max,
            step=step,
            density=density,
            title=title,
            y_lim=y_lim,
            include_under=include_under,
            accumulate=True,
            start_from=start_from,
            for_paper=self.for_paper,
            line_styles=line_styles,
            color_pallet=color_pallet,
        )

    def load_cache(self, path: Path) -> Any:
        return pickle.loads(path.read_bytes())

    def color_pallet(
        self,
    ):
        return "mako" if any(["ANCE" in label for label in self.labels]) else "rocket"

    def run(self) -> None:
        visualizer = DistributionVisualizer(self.damaged_start_at, self.damaged_until)
        ranking_shifts, nrses = [], []
        for result_path in self.result_pathes:
            ranking_shift, nrs, _, _ = visualizer.calc_shift_ranks(
                self.load_cache(result_path), is_intact=self.is_intact
            )
            ranking_shifts.append(ranking_shift)
            nrses.append(nrs)
        line_styles = self.line_styles(self.labels)

        base_save_path = Path(__file__).parent
        if self.is_intact:
            base_save_path = base_save_path.joinpath("sentence_intact")

        save_path = base_save_path.joinpath("damaged_ranking_shifts")

        save_path = base_save_path.joinpath("damaged_ranking_shifts_acc_line")
        self.plot_acc_shift_frequency_distr_line(
            visualizer,
            ranking_shifts,
            step=1,
            save_dir=save_path,
            line_styles=line_styles,
        )
        self.plot_acc_shift_frequency_distr_line(
            visualizer,
            ranking_shifts,
            hist_min=50,
            step=1,
            density=True,
            y_lim=15,
            include_under=False,
            save_dir=save_path,
            line_styles=line_styles,
            color_pallet=self.color_pallet(),
        )

        self.accum_freq_distr_table(
            visualizer,
            ranking_shifts,
            self.labels,
            hist_min=25,
            hist_max=200,
            step=25,
            density=True,
            include_under=True,
            save_dir=save_path,
        )


def parse_args() -> Namespace:
    parser = ArgumentParser(description="intact bm25_comparison")
    parser.add_argument("result_files", nargs="*")
    parser.add_argument(
        "--intact",
        action="store_true",
        default=False,
        help="True indicating passed result files ",
    )
    parser.add_argument(
        "--ignore-extra",
        action="store_true",
        default=False,
        help="True indicating passed result files ",
    )
    parser.add_argument(
        "--paper",
        action="store_true",
        default=False,
        help="True indicating output graphs for paper. (e.g. remove title)",
    )

    args, other = parser.parse_known_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    CompareRankshiftsDistribution(
        [Path(filepath) for filepath in args.result_files],
        intact=args.intact,
        ignore_extra=args.ignore_extra,
        for_paper=args.paper,
    ).run()
