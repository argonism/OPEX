import datetime
import json
import logging
import pickle
import configparser
from collections import defaultdict
from logging import getLogger
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple, Union
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import more_itertools

import gokart
import luigi
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.stem import PorterStemmer, WordNetLemmatizer
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy import stats

sys.path.append(str(Path(__file__).parent.parent))
from denserr.analyzer.damaged_analyzer import DamagedAnalyze
from denserr.analyzer.sentence_intact_analyzer import SentenceInstactAnalyze
from denserr.analyzer.distribution_visualizer import DistributionVisualizer
from denserr.retrieve import Retrieve
from denserr.dataset._base import PolarsCorpusLoader
from denserr.dataset.load_dataset import LoadDataset
from denserr.utils.to_markdown import MDFY
from denserr.utils.util import cache_dir, IterCacher, project_dir, write_json_to_file

from sentence_transformers import SentenceTransformer, util

logger = getLogger(__name__)


class SentenceSimilarity:
    def __init__(self, device: str = "cuda"):
        model_name = "all-mpnet-base-v2"
        self.model = SentenceTransformer(model_name, device=device)

    def similarity(self, sent: str, doc: str) -> float:
        query_embedding = self.model.encode(sent)
        passage_embedding = self.model.encode(doc)

        return util.dot_score(query_embedding, passage_embedding)

    def similarities_list(self, sent: List[str], doc: List[str]) -> List[float]:
        query_embedding = self.model.encode(sent)
        passage_embedding = self.model.encode(doc)

        cosine_scores = util.cos_sim(query_embedding, passage_embedding)

        scores = []
        for i in range(len(sent)):
            score = cosine_scores[i][i]
            scores.append(float(score))
        return scores


class PosIdentifier:
    def __init__(self, corpus: Optional[PolarsCorpusLoader] = None):
        self.corpus = corpus

    def damaged(self, docid: str, perturbation: str) -> Optional[int]:
        doc = self.corpus[docid]
        text = doc["text"]
        perturbation = perturbation.lower().replace("\n", "").replace(" ", "")
        sents = nltk.sent_tokenize(text)
        for i, sent in enumerate(sents):
            sent = sent.lower().replace("\n", "").replace(" ", "")
            if perturbation in sent:
                return i
        # print("(doc[text]:", text)
        # print("perturbation:", perturbation)
        return None

    def intact(self, perturbed: str, perturbation: str) -> Optional[int]:
        perturbed = perturbed
        perturbation = perturbation.lower().replace("\n", "").replace(" ", "")

        sents = nltk.sent_tokenize(perturbed)
        for i, sent in enumerate(sents):
            sent = sent.lower().replace("\n", "").replace(" ", "")
            if perturbation in sent:
                return i
        return None


class ANCEScore:
    def __init__(self):
        from denserr.model.ance import AnceTextEncoder

        self.ance_encoder = AnceTextEncoder()

    def score(self, query: str, docs: List[str]) -> List[float]:
        query_emb = self.ance_encoder.encode_queries([query])
        doc_emb = self.ance_encoder.encode_corpus(docs)


class ConfigGenerator(object):
    def __init__(self, base_config: Dict) -> None:
        self.base_config = base_config

    @classmethod
    def read_param_ini(
        cls, param_path: Path = Path("conf/param.ini"), key: str = "DenseErrConfig"
    ):
        config = configparser.ConfigParser()
        config.read(str(param_path))

        return config[key]

    def yield_setting(
        self, dataset_names: List[str], model_names: List[str]
    ) -> Generator[Dict, None, None]:
        base_config = self.base_config
        for dataset in self.dataset_names:
            base_config["dataset_name"] = dataset

            for model in self.model_names:
                base_config["model_name"] = model

                yield base_config


class WordOverwrap(object):
    def __init__(self) -> None:
        pass

    def text_to_normalized_tokens(self, text, use_stemming=True):
        text = text.lower()

        tokenizer = RegexpTokenizer(r"\w+")
        tokens = tokenizer.tokenize(text)

        stop_words = set(stopwords.words("english"))
        tokens = [token for token in tokens if token not in stop_words]

        if use_stemming:
            stemmer = PorterStemmer()
            tokens = [stemmer.stem(token) for token in tokens]
        else:
            lemmatizer = WordNetLemmatizer()
            tokens = [lemmatizer.lemmatize(token) for token in tokens]

        return tokens

    def added_sentence_word_overwrap(self, sentence: str, doc: str) -> float:
        sentence_tokens = set(self.text_to_normalized_tokens(sentence))
        doc_tokens = set(self.text_to_normalized_tokens(doc))
        overwrap_num = len([w for w in sentence if w in doc_tokens])
        union = len(sentence_tokens.union(doc_tokens))
        if union <= 0:
            return (0, 0, 0, 0)
        jaccard_distance = len(sentence_tokens.intersection(doc_tokens)) / union

        return overwrap_num, jaccard_distance, len(sentence_tokens), len(doc_tokens)


def parallel_analyze(
    i,
    part_result,
    retrieval_result,
    pos_identifier,
    skip_rs_lt: Optional[int] = None,
    skip_rs_ge: Optional[int] = None,
    is_intact: bool = False,
):
    sent_sim_calc = SentenceSimilarity(device=f"cuda:{i}")
    word_overwrap = WordOverwrap()
    part_result = list(part_result)

    similarities = []
    overwraps = []
    poses = []
    normed_sentence_lengthes = []
    normed_damaged_lens = []
    jaccard_list = []
    for qid, qid_result in tqdm(part_result):
        ranking = sorted(
            retrieval_result[qid].items(), key=lambda x: x[1], reverse=True
        )
        sents, docs = [], []
        for (
            new_rank,
            orig_rank,
            perturbed_score,
            damaged,
            perturbation,
        ) in qid_result:
            doc_id, score = ranking[orig_rank]
            rank_shift = new_rank - orig_rank if is_intact else orig_rank - new_rank
            if skip_rs_lt is not None and rank_shift < skip_rs_lt:
                continue
            if skip_rs_ge is not None and rank_shift >= skip_rs_ge:
                continue

            (
                num_overwrap,
                jaccard,
                sent_len,
                damaged_len,
            ) = word_overwrap.added_sentence_word_overwrap(perturbation, damaged)
            pos = (
                pos_identifier.intact(damaged, perturbation)
                if is_intact
                else pos_identifier.damaged(doc_id, perturbation)
            )
            poses.append(pos)
            if pos is None:
                print(f"pos not found! {orig_rank} -> {new_rank} ({qid}, {doc_id})")
            overwraps.append(num_overwrap)
            jaccard_list.append(jaccard)
            normed_sentence_lengthes.append(sent_len)
            normed_damaged_lens.append(damaged_len)
            sents.append(perturbation)
            docs.append(damaged)

        sims = sent_sim_calc.similarities_list(sents, docs)
        similarities += sims
    return (
        similarities,
        overwraps,
        [pos for pos in poses if pos is not None],
        normed_sentence_lengthes,
        normed_damaged_lens,
        jaccard_list,
    )


class RankshiftAnalyzer(object):
    def __init__(
        self,
        is_intact: bool = False,
        skip_rs_lt: Optional[int] = None,
        skip_rs_ge: Optional[int] = None,
        max_workers: int = 1,
    ):
        self.base_config = ConfigGenerator.read_param_ini()
        self.config_generator = ConfigGenerator(self.base_config)
        self.is_intact = is_intact
        self.skip_rs_lt = skip_rs_lt
        self.skip_rs_ge = skip_rs_ge
        self.max_workers = max_workers

    def set_config(self, config: Dict, config_key: str = "DenseErrConfig"):
        for k, v in config.items():
            luigi.configuration.get_config().set(config_key, k, str(v))

    def get_bm25_result(self, dataset_name: str):
        config = self.config_generator.read_param_ini()
        config["dataset_name"] = dataset_name
        config["model_name"] = "bm25"
        return self.get_damaged_result(config)

    def get_damaged_result(self, config: str):
        self.set_config(config)
        result = gokart.build(
            DamagedAnalyze(rerun=False),
            return_value=True,
            log_level=logging.ERROR,
        )
        return result

    def load_corpus(self, dataset_name: int) -> PolarsCorpusLoader:
        corpus, _, _ = gokart.build(
            LoadDataset(rerun=False),
            return_value=True,
            log_level=logging.ERROR,
        )
        return corpus

    def analyze(
        self,
        dataset_name: str,
        model_name: str,
        intact_context: str = "corpus",
    ) -> Dict[str, List[Union[float, int]]]:
        config = self.base_config
        config["dataset_name"] = dataset_name
        config["model_name"] = model_name
        config["perturb_context"] = intact_context
        self.set_config(config)
        visualizer = DistributionVisualizer(
            config["damaged_start_at"], config["damaged_until"]
        )

        analyzer = SentenceInstactAnalyze if self.is_intact else DamagedAnalyze
        result = gokart.build(
            analyzer(rerun=False),
            return_value=True,
            log_level=logging.INFO,
        )
        retrival_result = gokart.build(
            Retrieve(rerun=False),
            return_value=True,
            log_level=logging.INFO,
        )

        bm25_result = self.get_bm25_result(dataset_name)
        corpus = self.load_corpus(dataset_name).to_dict()
        pos_identifier = PosIdentifier(None if self.is_intact else corpus)
        retrieval_result = retrival_result
        evaluated_result = defaultdict(list)
        futures = []
        items = result.items()
        with ProcessPoolExecutor(
            max_workers=self.max_workers, mp_context=mp.get_context("spawn")
        ) as executor:
            for i, part_result in enumerate(
                more_itertools.divide(self.max_workers, items)
            ):
                logger.info(f"queing task at {i}")
                device = f"cuda:{i}"

                future = executor.submit(
                    parallel_analyze,
                    i,
                    part_result,
                    retrieval_result,
                    pos_identifier,
                    skip_rs_lt=self.skip_rs_lt,
                    skip_rs_ge=self.skip_rs_ge,
                    is_intact=self.is_intact,
                )
                futures.append(future)

        for future in tqdm(
            as_completed(futures), desc="waiting futures", total=len(futures)
        ):
            (
                part_similarities,
                part_overwraps,
                part_poses,
                part_normed_sentence_lengthes,
                part_normed_damaged_lens,
                part_jaccard_list,
            ) = future.result()
            evaluated_result["similarities"] += part_similarities
            evaluated_result["overwraps"] += part_overwraps
            evaluated_result["sentence poses"] += part_poses
            evaluated_result[
                "normed_sentence_lengthes"
            ] += part_normed_sentence_lengthes
            evaluated_result["normed_damaged_lens"] += part_normed_damaged_lens
            evaluated_result["jaccard_list"] += part_jaccard_list

        return evaluated_result


class StatsTest(object):
    def __init__(
        self, whole_case: Dict[str, List[float]], limited_case: Dict[str, List[float]]
    ) -> None:
        """
        Recieve two dict that has metric names as key and corresponding metrics values
        as values like
        {
            "similarities": [0.2, 0.32, ...] ,
            "overwarps": [0, 3, ...],
            "sentence poses": [4, 8, ...],
            "normed_sentence_lengthes": [149, 44, ...],
            "normed_damaged_lens": [12, 20, ...],
            "jaccard_list": [0.55, 0.4323, ...],
        }

        - Each metrics values list has length equal to the number of
          (whole/limited) case.
        - "whole_case" wounld be the all documents that is used in
          sentence (deletion/addition) analysis.
        - "limited_case" would be the documents that has experienced
          100 or more ranks shift in sentence (deletion/addition) analysis.
        """
        self.whole_case = whole_case
        self.limited_case = limited_case

    def test(self) -> Dict[str, float]:
        test_results = {}
        for metric in self.whole_case:
            if metric not in self.limited_case:
                raise ValueError(
                    f"metric {metric} is not in limited_case() ({self.limited_case.keys()})"
                )
            whole_values = self.whole_case[metric]
            limited_values = self.limited_case[metric]
            # two-sided t-test for independent samples
            test_result = stats.ttest_ind(whole_values, limited_values)
            test_results[metric] = {
                "statistic": test_result.statistic,
                "pvalue": test_result.pvalue,
            }

        return test_results


if __name__ == "__main__":

    def ave_result(result: Dict) -> None:
        ave_results = {}
        for label, values in result.items():
            # if len(values) <= 0:
            #     continue
            ave_results[f"ave_{label}"] = sum(values) / len(values)
        return ave_results

    def analyze(
        dataset, model, skip_rs_lt, skip_rs_ge, is_intact, intact_context, max_workers
    ):
        return RankshiftAnalyzer(
            skip_rs_lt=skip_rs_lt,
            skip_rs_ge=skip_rs_ge,
            is_intact=is_intact,
            max_workers=max_workers,
        ).analyze(
            dataset,
            model,
            intact_context=intact_context,
        )

    def merge_dicts(dict_list: List[Dict]) -> Dict:
        merged_dict = {key: [] for key in dict_list[0]}
        for dic in dict_list:
            for metric in dic:
                merged_dict[metric] += dic[metric]
        return merged_dict

    datasets = ["robust04", "msmarco-doc"]
    # datasets = ["robust04"]
    models = [
        "ance",
        "colbert",
        "deepct",
        "ptsplade",
    ]
    is_intact = False
    intact_context = "corpus"
    skip_rs_threshold = 100
    max_workers = 4

    extra = f"intact/{intact_context}" if is_intact else "damaged"

    output_dir = project_dir.joinpath("scripts", "AnalyzeRankshift", extra)
    cache_base_path = cache_dir.joinpath("RankshiftAnalyzer", extra)
    cacher = IterCacher(
        cache_base_path, cache_strategy="each_file", key_gen_method="hash"
    )

    all_result = defaultdict(lambda: defaultdict(dict))
    for dataset in datasets:
        print(f"## {dataset}")

        dataset_lt_results = []
        dataset_ge_results = []
        for model in models:
            print(f"### {model}")
            rs_lt_result_key = f"{dataset}_{model}_rs_<_{skip_rs_threshold}"
            rs_lt_result = cacher.load(rs_lt_result_key)
            if rs_lt_result is None:
                print(f"cache ({rs_lt_result_key}) not found")
                rs_lt_result = analyze(
                    dataset,
                    model,
                    None,
                    skip_rs_threshold,
                    is_intact,
                    intact_context,
                    max_workers,
                )
                cacher.cache(rs_lt_result, rs_lt_result_key)

            rs_ge_result_key = f"{dataset}_{model}_rs_>=_{skip_rs_threshold}"
            rs_ge_result = cacher.load(rs_ge_result_key)
            if rs_ge_result is None:
                print(f"cache ({rs_ge_result_key}) not found")
                rs_ge_result = analyze(
                    dataset,
                    model,
                    skip_rs_threshold,
                    None,
                    is_intact,
                    intact_context,
                    max_workers,
                )
                cacher.cache(rs_ge_result, rs_ge_result_key)

            # dataset_whole_results.append(whole_result)
            # dataset_rs100_results.append(rs100_result)

            # all_whole_result = merge_dicts(dataset_whole_results)
            # all_rs100_result = merge_dicts(dataset_rs100_results)
            # these result are dict that has structure like
            # (all_whole_result / all_rs100_result) = {
            #     "similarities": [0.2, 0.32, ...] ,
            #     "overwarps": [0, 3, ...],
            #     "sentence poses": [4, 8, ...],
            #     "normed_sentence_lengthes": [149, 44, ...],
            #     "normed_damaged_lens": [12, 20, ...],
            #     "jaccard_list": [0.55, 0.4323, ...],
            # }

            print(f"**{dataset} rs_lt_result result**")
            md_converter = MDFY(ave_result(rs_lt_result))
            md_table = md_converter.dict_to_md_table(
                transpose=True, key_label="metric", value_label="value"
            )
            print(md_table)
            write_json_to_file(
                all_result,
                output_dir.joinpath(dataset, f"{model}_rs_lt_result.json"),
            )

            print(f"**{dataset} rs_ge_result result**")
            md_converter = MDFY(ave_result(rs_ge_result))
            md_table = md_converter.dict_to_md_table(
                transpose=True, key_label="metric", value_label="value"
            )
            print(md_table)
            write_json_to_file(
                all_result,
                output_dir.joinpath(dataset, f"{model}_rs_ge_result.json"),
            )

            test_results = StatsTest(rs_lt_result, rs_ge_result).test()
            pvalues_table = {
                metric: t_res["pvalue"] for metric, t_res in test_results.items()
            }

            print("**Statistical test**")
            md_converter = MDFY(pvalues_table, precision=3)
            md_table = md_converter.dict_to_md_table(
                transpose=True, key_label="metric", value_label="pvalue"
            )
            print(md_table)

        # all_result[dataset][model]["Overall"] = ave_result(whole_result)
        # all_result[dataset][model]["Rank shift over 100"] = ave_result(rs100_result)
        # all_result[dataset][model]["test_results"] = test_results

        # write_json_to_file(
        #     [whole_result, rs100_result, test_results],
        #     output_dir.joinpath(f"{dataset}_{model}_{skip_rs_lt}.json"),
        # )
