import logging
import configparser
from collections import defaultdict
from logging import getLogger
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple, NamedTuple
import sys
import json

import gokart
import luigi
from tqdm import tqdm
from scipy import stats
import statsmodels.stats.multitest as multi

sys.path.append(str(Path(__file__).parent.parent))
from denserr.retrieve import EvaluatePerQuery
from denserr.utils.to_markdown import MdTable
from denserr.utils.util import cache_dir, IterCacher, project_dir, write_json_to_file
from scripts.analyze_rankshift import ConfigGenerator

logging.basicConfig(level=logging.INFO)
logger = getLogger(__name__)


class IRMetricsStatTester(object):
    def __init__(
        self, metrics: List[str] = ["ndcg", "_map", "recall", "precision"]
    ) -> None:
        self.base_config = ConfigGenerator.read_param_ini()

    def set_config(self, config: Dict, config_key: str = "DenseErrConfig") -> None:
        for k, v in config.items():
            luigi.configuration.get_config().set(config_key, k, str(v))

    def get_ir_metrics_result(self, config) -> Tuple:
        result = gokart.build(
            EvaluatePerQuery(rerun=False),
            return_value=True,
            log_level=logging.ERROR,
        )
        return result

    def get_model_metrics(self, dataset_name: str, model_name: str) -> Dict:
        config = self.base_config
        config["dataset_name"] = dataset_name
        config["model_name"] = model_name
        self.set_config(config)
        return self.get_ir_metrics_result(config)

    def test_one_pair_one_metrics(
        self, base_result: Dict, target_result: Dict
    ) -> NamedTuple:
        return stats.ttest_rel(base_result, target_result)

    def test_one_metrics(self, base_result: Dict, target_results: List[Dict]):
        models, pvalues, statistics, dfs = [], [], [], []
        for model_name, target_result in target_results:
            test_result = self.test_one_pair_one_metrics(base_result, target_result)
            models.append(model_name)
            pvalues.append(test_result.pvalue)
            statistics.append(test_result.statistic.tolist())
            dfs.append(test_result.df.tolist())
        return models, pvalues, statistics, dfs

    def holm_adjustments(self, pvalues) -> Tuple[List[bool], List[float]]:
        reject, pvals_corrected, alphac_sidak, alphac_bonf = multi.multipletests(
            pvalues, alpha=0.05, method="holm"
        )

        return reject.tolist(), pvals_corrected.tolist()

    def test_all_metrics(
        self, dataset_name: str, test_models: Dict[str, List[str]]
    ) -> Dict[str, Dict[str, Dict[str, str]]]:
        test_results = defaultdict(lambda: defaultdict(list))
        for base_model in test_models:
            print("base_model:", base_model)
            # Collect ir metrics evaluation results
            base_result_metrics = self.get_model_metrics(dataset_name, base_model)
            taget_results_metrics = defaultdict(list)
            for target_model in test_models[base_model]:
                target_result = self.get_model_metrics(dataset_name, target_model)
                for metric in target_result:
                    taget_results_metrics[metric].append(
                        (target_model, target_result[metric])
                    )

            for metric in base_result_metrics:
                # Paired T-test between base_model and target_models.
                base_result = base_result_metrics[metric]
                target_results = taget_results_metrics[metric]
                models, pvalues, statistics, dfs = self.test_one_metrics(
                    base_result, target_results
                )

                # Then, collet test result by ir metrics.
                for model, pvalue, statistic, df in zip(
                    models, pvalues, statistics, dfs
                ):
                    test_results[metric]["models"].append(model)
                    test_results[metric]["pvalues"].append(pvalue)
                    test_results[metric]["statistics"].append(statistic)
                    test_results[metric]["dfs"].append(df)

        # Holm adjustment and
        adjusted_result = {}
        for metric in test_results:
            pvalues = test_results[metric]["pvalues"]
            models = test_results[metric]["models"]
            rejects, adjusted_pvalues = self.holm_adjustments(pvalues)

            metrics_result = {
                model: {"pvalue": pvalue, "statistic": statistic, "df": df}
                for model, pvalue, statistic, df in zip(
                    models,
                    adjusted_pvalues,
                    test_results[metric]["statistics"],
                    test_results[metric]["dfs"],
                )
            }
            adjusted_result[metric] = metrics_result
            adjusted_result[metric]["reject"] = {
                model: reject for model, reject in zip(models, rejects)
            }

        return adjusted_result


def main():
    datasets = ["robust04", "msmarco-doc"]
    test_models = {
        "ptsplade": [
            "ptsplade-sent-parallel",
            "ptsplade-sent-w5-parallel",
            "ptsplade-sent-w10-parallel",
            "ptsplade-sent-w15-parallel",
            "ptsplade-sent-w20-parallel",
        ],
        "ance": [
            "ance-sent",
            "ance-sent-w5",
            "ance-sent-w10",
            "ance-sent-w15",
            "ance-sent-w20",
        ],
    }

    output_dir = project_dir.joinpath("scripts", "IRMetricsStatTest")
    output_dir.mkdir(parents=True, exist_ok=True)

    tester = IRMetricsStatTester()
    for dataset in datasets:
        output_path = output_dir.joinpath(f"{dataset}.json")
        print(f"## {dataset}")
        result = tester.test_all_metrics(dataset, test_models)
        # output_path.write_text(json.dumps(result, ensure_ascii=False))
        logger.info(f"writeout to {output_path}")

        pvalue_result = defaultdict(dict)
        for metric in result:
            print(f"### {metric}", "\n")
            metric_result = result[metric]
            for model in metric_result:
                if model == "reject":
                    markdown = MdTable(result[metric][model], precision=3)
                    print(markdown, "\n")
                else:
                    pvalue = result[metric][model]["pvalue"]
                    pvalue_result[metric][model] = pvalue
        markdown = MdTable(pvalue_result, precision=3)
        print(markdown, "\n")


if __name__ == "__main__":
    main()
