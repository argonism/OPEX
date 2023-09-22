from __future__ import annotations

import datetime
import math
import json
from collections import defaultdict
from pathlib import Path
from typing import (
    Dict,
    List,
    Callable,
    Union,
    Iterable,
    Any,
    Tuple,
    TypeVar,
    Iterator,
    Optional,
)
from more_itertools import windowed
import pickle
import hashlib

from tqdm import tqdm
import nltk
import pandas as pd
import more_itertools
import pyterrier as pt

project_dir = Path(__file__).parent.parent.parent
logs_dir = project_dir.joinpath("logs")
jsons_dir = logs_dir.joinpath("jsons")
cache_dir = project_dir.joinpath("cache")

index_base_path = project_dir.joinpath("index")


def now_log_dir(
    dir_name: str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
) -> Path:
    log_path = logs_dir.joinpath(dir_name)

    if not log_path.exists():
        log_path.mkdir(parents=True, exist_ok=True)

    return log_path


def write_json_to_file(
    json_encodable: dict,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(json_encodable, ensure_ascii=False, indent=2))


def writeout_json_to_log(
    json_encodable: dict,
    filename: str,
    dir_name: str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
) -> None:
    filename = f"{filename}.json"
    log_path = now_log_dir()

    log_path = log_path.joinpath(filename)
    log_path.write_text(json.dumps(json_encodable, ensure_ascii=False, indent=2))


def marge_files(file_pathes: List[Path], output_path: Path) -> int:
    write_count = 0
    with output_path.open("w") as fw:
        for file_path in tqdm(file_pathes, desc="file_pathes"):
            with file_path.open("r") as f:
                for line in tqdm(f):
                    fw.write(line)
                    write_count += 1
            # file_path.unlink()
    return write_count


def breakup_to_sentenses(
    return_type: str, window_size: int = 1
) -> Callable[[pd.DataFrame], Union[pd.DataFrame, List[Dict]]]:
    def _inner(inp: pd.DataFrame) -> Union[pd.DataFrame, List[Dict]]:
        new_rows = []
        for index, row in inp.iterrows():
            sentences = nltk.sent_tokenize(row["text"])
            for i, windowed_sentences in enumerate(windowed(sentences, window_size)):
                not_none_sentences = [
                    sent for sent in windowed_sentences if sent is not None
                ]
                if len(not_none_sentences) <= 0:
                    continue
                text = " ".join(not_none_sentences)
                columns_to_update = {"text": text, "docno": f'{row["docno"]}_{i}'}
                new_row = row.to_dict()
                new_row.update(columns_to_update)
                new_rows.append(new_row)

        if return_type == "df":
            return pd.DataFrame.from_records(new_rows)
        elif return_type == "list":
            return new_rows
        else:
            raise ValueError(f"Unknown return_type: {return_type}")

    return _inner


def aggregate_sentences(inp: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    # First, aggregate rows according to docno
    doc_rows: List[pd.Series] = []
    qid_docno_table: Dict[str, Dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for index, row in inp.iterrows():
        docno, sentid = row["docno"].rsplit("_", 1)
        qid_docno_table[row["qid"]][docno].append(row)

    # Then take max score among sentence for doc score
    agg_result: Dict[str, Dict[str, float]] = defaultdict(dict)
    for qid, doc_table in qid_docno_table.items():
        for docno in doc_table:
            rows = doc_table[docno]
            score = max([row["score"] for row in rows])
            agg_result[qid][docno] = score

    return agg_result


def simple_batching(
    transformer: pt.Transformer, batch_size: int, batch_for: str = "list"
) -> pt.Transformer:
    def batching_for_list(array: List[Dict]) -> pd.DataFrame:
        results = []
        for batch in tqdm(
            more_itertools.batched(array, batch_size),
            desc="simple batching",
            total=math.ceil(len(array) / batch_size),
        ):
            # for batch in more_itertools.batched(array, batch_size):
            batch_df = pd.DataFrame.from_records(batch)
            res = transformer.transform(batch_df)
            results.append(res)

        try:
            result_df = pd.concat(results)
        except ValueError as e:
            print(f"ERROR: {e}")
            print("array:", array)
            print(f"array length: {len(array)}")
            print("returning empty df")

            return pd.DataFrame()
        return result_df

    # def batching_for_df(df: pd.DataFrame) -> pd.DataFrame:
    #     num_batches = len(df) // batch_size + (1 if len(df) % batch_size else 0)
    #     for i in range(num_batches):
    #         # Extract the batch
    #         batch_df = df.iloc[i * batch_size : (i + 1) * batch_size]
    #         list_of_dfs.append(batch_df)

    if batch_for == "list":
        return batching_for_list
    else:
        raise Exception(f"Unknown batch_for param: {batch_for}")


T = TypeVar("T")


# Key Generation Strategies
class KeyGenerationStrategy:
    def generate_key(self, obj: Any) -> Any:
        raise NotImplementedError


class HashKeyGenerationStrategy(KeyGenerationStrategy):
    def generate_key(self, obj: Any) -> Any:
        return hashlib.sha256(pickle.dumps(obj)).hexdigest()


class SimpleKeyGenerationStrategy(KeyGenerationStrategy):
    def generate_key(self, obj: Any) -> Any:
        return obj


# Cache Load and Store Strategies
class CacheStrategy:
    def load(self, key: Any) -> Any:
        raise NotImplementedError

    def store(self, key: Any, value: Any) -> None:
        raise NotImplementedError


class FileCacheStrategy(CacheStrategy):
    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def load(self, key: Any) -> Any:
        cache_file = self.cache_dir / key
        if cache_file.exists():
            return pickle.loads(cache_file.read_bytes())
        return None

    def store(self, key: Any, value: Any) -> None:
        cache_file = self.cache_dir / key
        cache_file.write_bytes(pickle.dumps(value))


class MemoryCacheStrategy(CacheStrategy):
    def __init__(self, cache_path: Path) -> None:
        self.cache_path = cache_path
        self.cache = {}
        if self.cache_path.exists():
            self.cache = pickle.loads(self.cache_path.read_bytes())

    def load(self, key: Any) -> Any:
        return self.cache.get(key)

    def store(self, key: Any, value: Any) -> None:
        self.cache[key] = value
        self.cache_path.write_bytes(pickle.dumps(self.cache))


class IterCacher:
    def __init__(
        self,
        cache_dir: str,
        cache_strategy: str = "file",
        key_gen_method: str = "simple",
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.parent.mkdir(parents=True, exist_ok=True)

        # Set Key Generation Strategy
        if key_gen_method == "hash":
            self.key_gen = HashKeyGenerationStrategy()
        elif key_gen_method == "simple":
            self.key_gen = SimpleKeyGenerationStrategy()
        else:
            raise ValueError(f"Unsupported key_gen_method: {key_gen_method}")

        # Set Cache Strategy
        if cache_strategy == "file":
            self.cache_strategy = MemoryCacheStrategy(self.cache_dir)
        elif cache_strategy == "each_file":
            self.cache_strategy = FileCacheStrategy(self.cache_dir)
        else:
            raise ValueError(f"Unsupported cache_strategy: {cache_strategy}")

        self.last_cache_key = None

    def iter(self, iterable: Iterator[Any]) -> Iterator[Tuple["IterCacher", Any, Any]]:
        for obj in iterable:
            self.last_cache_key = self.key_gen.generate_key(obj)
            cached_value = self.cache_strategy.load(self.last_cache_key)
            yield self, obj, cached_value

    def cache(self, processed_obj: Any, key: Optional[Any] = None) -> None:
        # If a key is explicitly provided, use it. Else, fall back to last_cache_key.
        key_to_use = key if key is not None else self.last_cache_key
        self.cache_strategy.store(key_to_use, processed_obj)

    def load(self, key: Any) -> Any:
        return self.cache_strategy.load(key)
