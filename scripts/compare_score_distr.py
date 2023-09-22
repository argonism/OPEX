from typing import Generator, Optional, Union, Any, List
from pathlib import Path
from argparse import ArgumentParser, Namespace
import pickle
import sys
from logging import getLogger
from collections import defaultdict

import numpy as np

logger = getLogger(__name__)

sys.path.append(str(Path(__file__).parent.parent))
from denserr.analyzer.distribution_visualizer import DistributionVisualizer


class CompareScoreDistribution:
    def __init__(self, base_result_path: Path, target_result_path: Path) -> None:
        self.base_result_path = base_result_path
        self.target_result_path = target_result_path
        base_info, target_info = self.validate_same_setting()
        self.base_info = base_info
        self.target_info = target_info
        self.dataset_name = base_info["dataset"]
        self.base_model_name = base_info["model_name"]
        self.target_model_name = target_info["model_name"]

    def filepath_to_info(self, path: Path) -> dict:
        dataset = path.parent.stem
        path_stem = path.stem
        (
            model_name,
            _,
        ) = path_stem.split("_")
        return {
            "dataset": dataset,
            "model_name": model_name,
        }

    def validate_same_setting(self) -> tuple:
        logger.info("validating both cached results are under the same setting ...")
        base_info = self.filepath_to_info(self.base_result_path)
        target_info = self.filepath_to_info(self.target_result_path)
        for info_key in base_info:
            if info_key in ["model_name"]:
                continue
            logger.debug(f"validating: {info_key}")
            if not base_info[info_key] == target_info[info_key]:
                raise Exception(
                    f"setting does not match! {base_info[info_key]} !== {target_info[info_key]}"
                )
        logger.info("all setting are matched")
        return base_info, target_info

    def plot_scores_distr(
        self,
        visualizer: DistributionVisualizer,
        scores: List[List[float]],
        labels: List[str],
        save_dir: Path = Path(__file__).parent.joinpath("score_distr"),
    ) -> None:
        title = "\n".join(
            [
                f"{self.base_model_name} & {self.target_model_name} on {self.dataset_name} score distribution"
            ]
        )

        save_fig_path = save_dir.joinpath(
            f"{self.dataset_name}/{self.base_model_name}_{self.target_model_name}"
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

    def load_cache(self, path: Path) -> Any:
        return pickle.loads(path.read_bytes())

    def run(self) -> None:
        visualizer = DistributionVisualizer(0, 0)
        base_retrieval_result = visualizer.normalize_score_distr(
            self.load_cache(self.base_result_path), from_retrieval_result=True
        )
        target_retrieval_result = visualizer.normalize_score_distr(
            self.load_cache(self.target_result_path), from_retrieval_result=True
        )

        self.plot_scores_distr(
            visualizer,
            [base_retrieval_result, target_retrieval_result],
            [self.base_model_name, self.target_model_name],
        )


def parse_args() -> Namespace:
    parser = ArgumentParser(description="intact bm25_comparison")
    parser.add_argument("base_result")
    parser.add_argument("target_result")
    args, other = parser.parse_known_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    CompareScoreDistribution(Path(args.base_result), Path(args.target_result)).run()
