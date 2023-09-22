import datetime
import json
import logging
import pickle
import configparser
from collections import defaultdict
from logging import getLogger
from pathlib import Path
from typing import Dict, Generator, List
import sys

import gokart
import luigi
import pandas as pd
import nltk
from tqdm import tqdm
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent.parent))
from denserr.dataset.load_dataset import LoadDataset
from denserr.analyzer.distribution_visualizer import DistributionVisualizer
from denserr.utils.util import IterCacher, cache_dir

import matplotlib as mpl

mpl.rcParams["pdf.use14corefonts"] = True

logger = getLogger(__name__)


def read_param_ini(
    param_path: Path = Path("conf/param.ini"), key: str = "DenseErrConfig"
):
    config = configparser.ConfigParser()
    config.read(str(param_path))

    return config[key]


def yield_setting(retrieval_type: str) -> Generator[Dict, None, None]:
    config = read_param_ini()

    # datasets = ["robust04"]
    datasets = ["robust04", "msmarco-doc"]
    models = [
        # "bm25",
        # "ance",
        # "dpr",
        # "colbert",
        # "deepct",
        "bm25",
    ]

    for dataset in datasets:
        config["dataset_name"] = dataset

        for model in models:
            config["model_name"] = model

            yield config


def set_config(config: Dict, config_key: str = "DenseErrConfig"):
    for k, v in config.items():
        luigi.configuration.get_config().set(config_key, k, str(v))


def experiment() -> None:
    output_dir = Path(__file__).parent.joinpath("results")
    now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_dir.joinpath(now_str)

    visualizer = DistributionVisualizer(0, 100)
    cache_base_path = cache_dir.joinpath("doc_length")
    cacher = IterCacher()

    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    retrieval_type = "full"
    config_generator = yield_setting(retrieval_type)
    results = defaultdict(list)
    for config in config_generator:
        print(config)
        model_name = config["model_name"]
        dataset_name = config["dataset_name"]
        set_config(config)
        result = gokart.build(
            LoadDataset(rerun=False),
            return_value=True,
            log_level=logging.INFO,
        )
        results[dataset_name].append((model_name, result))

    for dataset_name in results:
        print(f"### {dataset_name}")
        for model_name, (corpus_loader, _, _) in results[dataset_name]:
            sentence_lengthes = []
            for doc in tqdm(corpus_loader):
                sents = nltk.sent_tokenize(doc["text"])
                sentence_lengthes.append(len(sents))

            ave_length = sum(sentence_lengthes) / len(sentence_lengthes)

            print(
                f"{model_name}: ave sentence length "
                + f"{ave_length}"
                + f"sentence like this: {sents[0]}"
            )

            file_name = f"{dataset_name}_{model_name}"
            save_dir = Path(__file__).parent.joinpath("sentence_length_dist")
            acc_save_path = save_dir.joinpath(f"{file_name}_acc.png")
            hist_save_path = save_dir.joinpath(f"{file_name}_hist.png")
            # all_save_path = save_dir.joinpath(f"{file_name}_all.png")

            title = f"{dataset_name} {model_name}. average sentence num: {ave_length}"

            # fig = plt.figure(figsize=(24, 7))
            # ax = fig.add_subplot(1, 1, 1)
            visualizer.plot_acc_ranking_shift_line(
                [sentence_lengthes],
                acc_save_path,
                labels=[dataset_name],
                step=1,
                hist_min=0,
                hist_max=40,
                title=title,
                grayout_first=False,
                for_paper=True,
                # fig=fig,
                # ax=ax,
            )

            visualizer.plot_shift_frequency_distr(
                [sentence_lengthes],
                hist_save_path,
                labels=[dataset_name],
                step=1,
                hist_min=0,
                hist_max=40,
                title=title,
                grayout_first=False,
                for_paper=True,
                # fig=fig,
                # ax=ax,
                xlabel="Sentence number",
                ylabel="Proportion",
            )
            # Path(all_save_path).parent.mkdir(parents=True, exist_ok=True)
            # fig.savefig(all_save_path)
            # logger.info(f"All-fig saved at : {all_save_path}")

            # fig = plt.figure(figsize=(24, 7))

            # plt.xlabel("sentence nums in doc")
            # plt.ylabel("frequency")
            # plt.title(
            #     f"{dataset_name} {model_name}. average sentence num: {ave_length}"
            # )

            # plt.hist(sentence_lengthes, bins=100, range=(0, 60))
            # fig.savefig(str(save_path))


if __name__ == "__main__":
    experiment()
