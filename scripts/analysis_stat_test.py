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
from denserr.analyzer.damaged_analyzer import DamagedAnalyze
from denserr.analyzer.sentence_intact_analyzer import SentenceInstactAnalyze
from denserr.analyzer.distribution_visualizer import DistributionVisualizer
from denserr.utils.to_markdown import MdTable
from denserr.utils.util import cache_dir, IterCacher, project_dir, write_json_to_file
from scripts.analyze_rankshift import ConfigGenerator

logging.basicConfig(level=logging.INFO)
logger = getLogger(__name__)


class SentenceAnalysisStatTester(object):
    def __init__(
        self,
        dataset_name: str,
        is_intact: bool = False,
        intact_perturb_context="corpus",
    ) -> None:
        self.base_config = ConfigGenerator.read_param_ini()
        self.dataset_name = dataset_name
        self.is_intact = is_intact
        self.intact_perturb_context = intact_perturb_context
        self.visualizer = DistributionVisualizer(
            self.base_config["damaged_start_at"], self.base_config["damaged_until"]
        )

    def set_config(self, config: Dict, config_key: str = "DenseErrConfig") -> None:
        for k, v in config.items():
            luigi.configuration.get_config().set(config_key, k, str(v))

    def get_analysis_result(self, config) -> Tuple:
        if self.is_intact:
            return gokart.build(
                SentenceInstactAnalyze(rerun=False),
                return_value=True,
                log_level=logging.ERROR,
            )

        return gokart.build(
            DamagedAnalyze(rerun=False),
            return_value=True,
            log_level=logging.ERROR,
        )

    def get_model_analysis_result(self, model_name: str) -> Dict:
        config = self.base_config
        config["dataset_name"] = self.dataset_name
        config["model_name"] = model_name
        if self.is_intact:
            config["perturb_context"] = self.intact_perturb_context
            config["perturb_position"] = "random"
        self.set_config(config)
        return self.get_analysis_result(config)

    def calc_rank_shifts(self, model_name: str) -> List[int]:
        analysis_result = self.get_model_analysis_result(model_name)
        (
            ranking_shifts,
            normalized_ranking_shifts,
            new_ranks,
            orig_ranks,
        ) = self.visualizer.calc_shift_ranks(analysis_result)
        return ranking_shifts

    def calc_contingency_row(
        self, rank_shifts: List[float], threshold: int
    ) -> List[int]:
        rank_shift_over_threshold = 0
        for rank_shift in rank_shifts:
            if rank_shift >= threshold:
                rank_shift_over_threshold += 1
        return [len(rank_shifts) - rank_shift_over_threshold, rank_shift_over_threshold]

    def gen_contingency_table(
        self,
        base_model: str,
        target_model: str,
        shift_threshold: int,
    ) -> List[List[float]]:
        base_rank_shift = self.calc_rank_shifts(base_model)
        base_contingency = self.calc_contingency_row(
            base_rank_shift, threshold=shift_threshold
        )

        target_rank_shift = self.calc_rank_shifts(target_model)
        target_contingency = self.calc_contingency_row(
            target_rank_shift, threshold=shift_threshold
        )

        return [base_contingency, target_contingency]

    def holm_adjustments(self, pvalues) -> Tuple[List[bool], List[float]]:
        reject, pvals_corrected, alphac_sidak, alphac_bonf = multi.multipletests(
            pvalues, alpha=0.05, method="holm"
        )

        return reject.tolist(), pvals_corrected.tolist()

    def test_sentence_analysis(
        self,
        test_models: Dict[str, List[str]],
        shift_threshold: int,
        with_holm_adjust: bool = True,
    ) -> Dict[str, Dict[str, Dict[str, str]]]:
        test_results: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(
            lambda: defaultdict(dict)
        )
        for base_model in test_models:
            for target_model in test_models[base_model]:
                """
                Create contingency_table for each (BM25, LLM-based IR model) pairs like this
                     | RS < threshold | RS >= threshold
                -------------------------------------------
                BM25 |      49984     |       16
                ANCE |      48729     |      1271

                contingency_table = [[49984, 16], [48729, 1271]]
                """

                contingency_table = self.gen_contingency_table(
                    base_model, target_model, shift_threshold
                )
                test_result = stats.chi2_contingency(contingency_table)
                test_results[target_model] = {
                    "statistic": test_result.statistic,
                    "pvalue": test_result.pvalue,
                    "dof": test_result.dof,
                    "expected_freq": test_result.expected_freq.tolist(),
                }

        if with_holm_adjust:
            models = []
            pvalues = []
            for target_model in test_results:
                models.append(target_model)
                pvalues.append(test_results[target_model]["pvalue"])

            print(f"with_holm_adjust with {len(pvalues)} pvalues")
            rejects, adjusted_pvalues = self.holm_adjustments(pvalues)
            for model, pvalue, reject in zip(models, adjusted_pvalues, rejects):
                test_results[model]["pvalue"] = pvalue
                test_results[model]["reject"] = reject

        return test_results


def main():
    dataset_test_models = {
        "robust04": {
            "ance": [
                "ance-sent",
                "ance-sent-w5",
                "ance-sent-w10",
                "ance-sent-w15",
                "ance-sent-w20",
            ],
            "ptsplade": [
                "ptsplade-sent-parallel",
                "ptsplade-sent-w5-parallel",
                "ptsplade-sent-w10-parallel",
                "ptsplade-sent-w15-parallel",
                "ptsplade-sent-w20-parallel",
            ],
        },
        "msmarco-doc": {
            "ance": [
                "ance-sent",
                "ance-sent-w5",
                "ance-sent-w10",
                "ance-sent-w15",
                "ance-sent-w20",
            ],
            "ptsplade": [
                "ptsplade-sent-parallel",
                "ptsplade-sent-w5-parallel",
                "ptsplade-sent-w10-parallel",
                "ptsplade-sent-w15-parallel",
                "ptsplade-sent-w20-parallel",
            ],
        },
    }

    intact_context = "corpus"
    thresholds = [50, 100, 150]

    for is_intact in [False]:
        analysis_method = f"Intact-{intact_context}" if is_intact else "Damaged"

        output_dir = project_dir.joinpath(
            "scripts", "SentenceAnalysisStatTest", analysis_method
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"# {analysis_method}: {output_dir}")

        for dataset in dataset_test_models:
            for threshold in thresholds:
                print(f"## {dataset}-{threshold}")
                test_models = dataset_test_models[dataset]
                tester = SentenceAnalysisStatTester(dataset, is_intact=is_intact)

                output_path = output_dir.joinpath(f"{dataset}-{threshold}.json")
                result = tester.test_sentence_analysis(
                    test_models, threshold, with_holm_adjust=True
                )
                output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
                logger.info(f"writeout to {output_path}")

                pvalue_result = defaultdict(dict)
                for target_model in result:
                    target_test_result = result[target_model]
                    print(f"{target_model} pvalue: {target_test_result['pvalue']:.3f}")
                    print(f"{target_model} reject: {target_test_result['reject']}")

                markdown = MdTable(result, precision=3)
                print(markdown, "\n")


if __name__ == "__main__":
    main()
