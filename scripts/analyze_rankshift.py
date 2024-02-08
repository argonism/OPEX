import datetime
import json
import logging
import pickle
import configparser
from collections import defaultdict
from logging import getLogger
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple, Union, Any
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
from sklearn.preprocessing import minmax_scale
from mdfy import MdTable

sys.path.append(str(Path(__file__).parent.parent))
from denserr.analyzer.damaged_analyzer import DamagedAnalyze
from denserr.analyzer.sentence_intact_analyzer import SentenceInstactAnalyze
from denserr.analyzer.distribution_visualizer import DistributionVisualizer
from denserr.retrieve import Retrieve
from denserr.dataset._base import PolarsCorpusLoader
from denserr.dataset.load_dataset import LoadDataset
from denserr.model.deepct import DeepctRetriever

# from denserr.utils.to_markdown import MDFY
from denserr.utils.util import cache_dir, IterCacher, project_dir, write_json_to_file

from sentence_transformers import SentenceTransformer, util

logger = getLogger(__name__)


class CompareElement(object):
    element_name = "compare_element"

    def calc(
        self,
        docids: List[str] = None,
        docs: List[str] = None,
        query: str = None,
        sents: List[str] = None,
        positions: List[int] = None,
        qrels: List[int] = None,
    ) -> List[Any]:
        ...

    def run(self, **kwargs):
        return self.calc(**kwargs)


class SentenceSimilarity(CompareElement):
    element_name = "sentence_sim"

    def __init__(self, device: str = "cuda"):
        model_name = "all-mpnet-base-v2"
        self.model = SentenceTransformer(model_name, device=device)

    def calc(self, *, docs: List[str], sents: List[str], **kwargs) -> List[Any]:
        query_embedding = self.model.encode(sents)
        passage_embedding = self.model.encode(docs)

        cosine_scores = util.cos_sim(query_embedding, passage_embedding)

        scores = []
        for i in range(len(sents)):
            score = cosine_scores[i][i]
            scores.append(float(score))
        return scores


class PosIdentifier(CompareElement):
    element_name = "pos"

    def __init__(self, corpus: Optional[PolarsCorpusLoader], is_intact: bool) -> None:
        self.corpus = corpus
        self.is_intact = is_intact

    def damaged(self, docid: str, perturbation: str) -> Optional[int]:
        doc = self.corpus[docid]
        text = doc["text"]
        perturbation = perturbation.lower().replace("\n", "").replace(" ", "")
        sents = nltk.sent_tokenize(text)
        for i, sent in enumerate(sents):
            sent = sent.lower().replace("\n", "").replace(" ", "")
            if perturbation in sent:
                return i
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

    def calc(self, positions: List[str], **kwargs) -> List[Any]:
        return positions


class QuerySentScore(CompareElement):
    element_name = "qs_score"

    def __init__(
        self, model_name: str, dataset_name: str, device: str = "cuda"
    ) -> None:
        from denserr.model.load_model import LoadRetriever

        self.model = LoadRetriever(dataset_name, model_name).load_retriever(
            device=device
        )

        if "deepct" in model_name:
            if not dataset_name == self.model.dataset_name:
                self.model.set_params(dataset_name)
                self.model._textscorer = None
                logger.info("DeepCT reinitialize")
                return

    @classmethod
    def preprocess(cls, lt, ge) -> Tuple:
        all_scores = lt[cls.element_name] + ge[cls.element_name]
        min_, max_ = min(all_scores), max(all_scores)
        norm_lt = map(lambda e: (e - min_) / (max_ - min_), lt[cls.element_name])
        norm_ge = map(lambda e: (e - min_) / (max_ - min_), ge[cls.element_name])
        lt[cls.element_name] = list(norm_lt)
        ge[cls.element_name] = list(norm_ge)
        return lt, ge

    def calc(self, query: str, sents: List[str], **kwargs) -> List[Any]:
        if hasattr(self.model, "batch_single_doc_score"):
            queries = [query] * len(sents)
            scores = self.model.batch_single_doc_score(queries, sents)
            scores = list(scores)
        else:
            scores = []
            for sent in sents:
                score = self.model.single_doc_score(query, sent)
                scores.append(score)
        return scores


def text_to_normalized_tokens(text, use_stemming=True):
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


class JaccardDistance(CompareElement):
    element_name = "jaccard_dist"

    def calc(self, docs: List[str], sents: List[str], **kwargs) -> List[Any]:
        jaccard_dists = []
        for doc, sentence in zip(docs, sents):
            sentence_tokens = set(text_to_normalized_tokens(sentence))
            doc_tokens = set(text_to_normalized_tokens(doc))

            union = len(sentence_tokens.union(doc_tokens))
            if union <= 0:
                return (0, 0, 0, 0)
            jaccard_distance = len(sentence_tokens.intersection(doc_tokens)) / union
            jaccard_dists.append(jaccard_distance)

        return jaccard_dists


class WordOverwrap(CompareElement):
    element_name = "word_overwrap"

    def added_sentence_word_overwrap(self, sentence: str, doc: str) -> float:
        sentence_tokens = set(text_to_normalized_tokens(sentence))
        doc_tokens = set(text_to_normalized_tokens(doc))
        overwrap_num = len([w for w in sentence if w in doc_tokens])

        return overwrap_num

    def calc(self, docs: List[str], sents: List[str], **kwargs) -> List[Any]:
        return [
            self.added_sentence_word_overwrap(sent, doc)
            for doc, sent in zip(docs, sents)
        ]


class AddedSentencesLength(CompareElement):
    element_name = "normed_sentence_lengthes"

    def calc(self, sents: List[str], **kwargs) -> List[Any]:
        return [len(set(text_to_normalized_tokens(sent))) for sent in sents]


class DocsLength(CompareElement):
    element_name = "normed_damaged_lens"

    def calc(self, docs: List[str], **kwargs) -> List[Any]:
        return [len(set(text_to_normalized_tokens(doc))) for doc in docs]


class CompareQrels(CompareElement):
    element_name = "qrel"

    def calc(self, qrels: List[str], **kwargs) -> List[Any]:
        return qrels


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


class CompareElementManager(object):
    def __init__(
        self, compare_elements: List[CompareElement], qrels: Dict[str, Dict[str, int]]
    ) -> None:
        self.compare_elements = compare_elements
        self.qrels = qrels

    def get_qrel_if_exist_else_0(self, qid, docid) -> int:
        if qid in self.qrels and docid in self.qrels[qid]:
            return self.qrels[qid][docid]
        else:
            return 0

    def execute(
        self,
        part_result,
        retrieval_result,
        queries_table: Dict[str, str],
        skip_rs_lt: Optional[int] = None,
        skip_rs_ge: Optional[int] = None,
        is_intact: bool = False,
        min_qrel: int = 0,
    ):
        execute_result = defaultdict(list)
        for qid, qid_result in tqdm(part_result, "comp_elem calculation"):
            query = queries_table[qid]
            ranking = sorted(
                retrieval_result[qid].items(), key=lambda x: x[1], reverse=True
            )
            sents, docs, docids, positions, qrels = [], [], [], [], []
            for (
                new_rank,
                orig_rank,
                perturbed_score,
                damaged,
                perturbation,
                position,
            ) in qid_result:
                doc_id, score = ranking[orig_rank]
                qrel = self.get_qrel_if_exist_else_0(qid, doc_id)
                rank_shift = new_rank - orig_rank if is_intact else orig_rank - new_rank
                if skip_rs_lt is not None and rank_shift < skip_rs_lt:
                    continue
                if skip_rs_ge is not None and rank_shift >= skip_rs_ge:
                    continue
                if qrel < min_qrel:
                    continue

                sents.append(perturbation)
                docs.append(damaged)
                docids.append(doc_id)
                positions.append(position)
                qrels.append(qrel)

            if len(sents) > 0:
                for comp_elem in self.compare_elements:
                    results = comp_elem.run(
                        docids=docids,
                        docs=docs,
                        query=query,
                        sents=sents,
                        positions=positions,
                        qrels=qrels,
                    )
                    execute_result[comp_elem.element_name] += results

        return execute_result


def parallel_analyze(
    i,
    part_result,
    retrieval_result,
    pos_identifier,
    model_name: str,
    dataset_name: str,
    queries_table: Dict[str, str],
    qrels: Dict[str, Dict[str, float]],
    skip_rs_lt: Optional[int] = None,
    skip_rs_ge: Optional[int] = None,
    is_intact: bool = False,
    min_qrel: int = 0,
):
    cuda_device = f"cuda:{i}"
    compare_elements = [
        SentenceSimilarity(device=cuda_device),
        JaccardDistance(),
        QuerySentScore(model_name, dataset_name, device=cuda_device),
        AddedSentencesLength(),
        pos_identifier,
        CompareQrels(),
        DocsLength(),
    ]
    manager = CompareElementManager(compare_elements, qrels)

    part_result = list(part_result)
    return manager.execute(
        part_result,
        retrieval_result,
        queries_table,
        skip_rs_lt,
        skip_rs_ge,
        is_intact,
        min_qrel,
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

    def load_dataset(self, dataset_name: int) -> Tuple[PolarsCorpusLoader, Dict, Dict]:
        corpus, queries, qrels = gokart.build(
            LoadDataset(rerun=False),
            return_value=True,
            log_level=logging.ERROR,
        )
        return corpus, queries, qrels

    def analyze(
        self,
        dataset_name: str,
        model_name: str,
        intact_context: str = "corpus",
        min_qrel: int = 0,
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
        corpus, queries, qrels = self.load_dataset(dataset_name)
        corpus = corpus.to_dict()
        pos_identifier = PosIdentifier(corpus, self.is_intact)
        retrieval_result = retrival_result
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
                    model_name,
                    dataset_name,
                    queries,
                    qrels,
                    skip_rs_lt=self.skip_rs_lt,
                    skip_rs_ge=self.skip_rs_ge,
                    is_intact=self.is_intact,
                    min_qrel=min_qrel,
                )
                futures.append(future)

        evaluated_result = defaultdict(list)
        for future in as_completed(futures):
            result = future.result()
            for k, v in result.items():
                evaluated_result[k] += v

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

    def concat_dict_lists(dict_list: List[Dict]) -> Dict:
        merged_dict = defaultdict(list)
        for dic in dict_list:
            for metric in dic:
                merged_dict[metric] += dic[metric]
        return merged_dict

    datasets = ["robust04", "msmarco-doc", "dl19-doc", "dl20-doc"]
    models = [
        "ance",
        "colbert",
        "deepct",
        "ptsplade",
    ]
    is_intact = False
    intact_context = "corpus"
    skip_rs_threshold = 100
    min_qrel = 0
    max_workers = 1

    analysis_setting = f"intact/{intact_context}" if is_intact else "damaged"
    extra = f"{skip_rs_threshold}_{min_qrel}"

    output_dir = project_dir.joinpath(
        "scripts", "AnalyzeRankshift", analysis_setting, extra
    )
    logger.info(f"output dir: {output_dir}")

    all_lt_results, all_ge_results = [], []
    for dataset in datasets:
        print(f"## {dataset}\n")
        dataset_output_path = output_dir.joinpath(dataset)

        model_lt_results = []
        model_ge_results = []
        for model in models:
            model_output_path = dataset_output_path.joinpath(model)
            rs_lt_result_path = model_output_path.joinpath("rs_lt_result.json")
            rs_ge_result_path = model_output_path.joinpath("rs_ge_result.json")

            if rs_lt_result_path.exists():
                rs_lt_result = json.loads(rs_lt_result_path.read_text())
            else:
                rs_lt_result = analyze(
                    dataset,
                    model,
                    None,
                    skip_rs_threshold,
                    is_intact,
                    intact_context,
                    max_workers,
                )

            if rs_ge_result_path.exists():
                rs_ge_result = json.loads(rs_ge_result_path.read_text())
            else:
                rs_ge_result = analyze(
                    dataset,
                    model,
                    skip_rs_threshold,
                    None,
                    is_intact,
                    intact_context,
                    max_workers,
                )

            rs_lt_result, rs_ge_result = QuerySentScore.preprocess(
                rs_lt_result, rs_ge_result
            )

            write_json_to_file(rs_lt_result, rs_lt_result_path)
            model_lt_results.append(rs_lt_result)

            write_json_to_file(rs_ge_result, rs_ge_result_path)
            model_ge_results.append(rs_ge_result)

        dataset_lt_results = concat_dict_lists(model_lt_results)
        write_json_to_file(
            dataset_lt_results,
            output_dir.joinpath(dataset, "rs_ge_result.json"),
        )
        dataset_ge_results = concat_dict_lists(model_ge_results)
        write_json_to_file(
            dataset_ge_results,
            output_dir.joinpath(dataset, "rs_ge_result.json"),
        )

        ave_dataset_lt_results = ave_result(dataset_lt_results)
        write_json_to_file(
            ave_dataset_lt_results,
            output_dir.joinpath(dataset, "ave_rs_lt_result.json"),
        )

        ave_dataset_ge_results = ave_result(dataset_ge_results)
        write_json_to_file(
            ave_dataset_ge_results,
            output_dir.joinpath(dataset, "ave_rs_ge_result.json"),
        )

        labels = ["metrics", f"RS < {skip_rs_threshold}", f"{skip_rs_threshold} ≤ RS"]
        results = [ave_dataset_lt_results, ave_dataset_ge_results]
        md_table = MdTable(results, labels=labels, transpose=True, precision=6)
        print(md_table, "\n")

        test_results = StatsTest(dataset_lt_results, dataset_ge_results).test()
        pvalues_table = {
            metric: t_res["pvalue"] for metric, t_res in test_results.items()
        }

        print("**Statistical test**\n")
        md_table = MdTable(pvalues_table, transpose=True, precision=4)

        print(md_table)
