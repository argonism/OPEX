import datetime
import sys
import json
import logging
import pickle
import configparser
from collections import defaultdict
from logging import getLogger
from pathlib import Path
from typing import Dict, Generator, List

import gokart
import luigi
import numpy as np
import pandas as pd

from denserr.retrieve import Evaluate
from denserr.analyzer.damaged_analyzer import DamagedAnalyze
from denserr.analyzer.distribution_visualizer import DistributionVisualizer

from scripts.compare_ranking_shifts import (
    MODEL_NAME_PATCH_TABLE,
)

logger = getLogger(__name__)


def read_param_ini(
    param_path: Path = Path("conf/param.ini"), key: str = "DenseErrConfig"
):
    config = configparser.ConfigParser()
    config.read(str(param_path))

    return config[key]


def set_config(config: Dict, config_key: str = "DenseErrConfig"):
    for k, v in config.items():
        luigi.configuration.get_config().set(config_key, k, str(v))


def acc_damaged_metric(
    damaged_result,
    at: int = 100,
    damaged_start_at: int = 100,
    damaged_until: int = 300,
    is_intact: bool = False,
):
    visualizer = DistributionVisualizer(damaged_start_at, damaged_until)
    ranking_shift, nrs, _, _ = visualizer.calc_shift_ranks(
        damaged_result, is_intact=is_intact
    )
    freq_distr, bin_edges = visualizer.histgramize(
        ranking_shift,
        hist_max=damaged_until,
        hist_min=-damaged_until,
        step=25,
        density=True,
        include_under=True,
    )
    acc_freq_distr = np.cumsum(freq_distr)
    for i, (freq, edge) in enumerate(zip(freq_distr, bin_edges[:-1])):
        if edge == at:
            return acc_freq_distr, i
    raise Exception(f"Maybe at={at} is not included in bin_edges ({bin_edges}).")


def yield_setting(retrieval_type: str) -> Generator[Dict, None, None]:
    config = read_param_ini()

    datasets = ["msmarco-doc"]
    # datasets = ["robust04"]
    # datasets = ["robust04", "msmarco-doc"]
    models = [
        # "bm25",
        "ance",
        "ance-sent",
        # "ance-sent-w2",
        # "ance-sent-w3",
        # "ance-sent-w4",
        "ance-sent-w5",
        "ance-sent-w10",
        "ance-sent-w15",
        # "ance-sent-w20",
        # "dpr",
        # "colbert",
        # "colbert-sent",
        # "deepct",
        # "deepct-sent",
        "ptsplade",
        # "ptsplade-parallel",
        # "ptsplade-sent",
        "ptsplade-sent-parallel",
        # "ptsplade-sent-w2-parallel",
        # "ptsplade-sent-w3-parallel",
        "ptsplade-sent-w5-parallel",
        "ptsplade-sent-w10-parallel",
        "ptsplade-sent-w15-parallel",
        "ptsplade-sent-w20-parallel",
        # "ptsplade-parallel-500",
        # "ptsplade-parallel-10000",
    ]

    for dataset in datasets:
        config["dataset_name"] = dataset

        for model in models:
            config["model_name"] = model
            if dataset == "msmarco-doc" and model == "bm25":
                config["model_name"] = "bm25-msmarco-doc"

            yield config


METRICS_TO_USE = ["NDCG@10", "NDCG@100", "MAP@10", "MAP@100", "Recall@10", "Recall@100"]


def experiment() -> None:
    output_dir = Path(__file__).parent.joinpath("results")
    now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_dir.joinpath(now_str)

    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    retrieval_type = "full"
    config_generator = yield_setting(retrieval_type)
    eval_results = defaultdict(list)
    damaged_results = defaultdict(dict)
    for config in config_generator:
        print(config)
        model_name = config["model_name"]
        for pattern, patch in MODEL_NAME_PATCH_TABLE.items():
            if pattern in model_name:
                model_name = model_name.replace(pattern, patch)
        dataset_name = config["dataset_name"]
        set_config(config)
        eval_result = gokart.build(
            Evaluate(rerun=False),
            return_value=True,
            log_level=logging.INFO,
        )
        eval_results[dataset_name].append((model_name, eval_result))

        damaged_result = gokart.build(
            DamagedAnalyze(rerun=False),
            return_value=True,
            log_level=logging.INFO,
        )
        acc_freq_distr, metric_index = acc_damaged_metric(damaged_result, 100)
        damaged_results[dataset_name][model_name] = acc_freq_distr[metric_index]

    for dataset_name in eval_results:
        print(f"### {dataset_name}")
        _, top_eval_result = eval_results[dataset_name][0]
        # eval_headers = list(top_eval_result.keys())
        eval_headers = list(METRICS_TO_USE)
        # misc_headers = ["damaged"]
        # headers = eval_headers + misc_headers
        headers = eval_headers
        print(f'||{"|".join(headers)}|')
        # print("|-" * (len(headers) + len(misc_headers)) + "|")
        print("|-" * (len(headers) + 1) + "|")
        for model_name, eval_result in eval_results[dataset_name]:
            # eval_rows = [f"{e:.3f}" for e in eval_result.values()]
            eval_rows = [f"{eval_result[metric]:.3f}" for metric in METRICS_TO_USE]
            # misc_rows = [f"{damaged_results[dataset_name][model_name]:.3f}"]
            rows = eval_rows
            # rows = eval_rows + misc_rows
            print(f"|{model_name}|" + "|".join(rows) + "|")


if __name__ == "__main__":
    experiment()
