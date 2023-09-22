import math
from logging import getLogger
from pathlib import Path
import json
import csv
from typing import Generator, Optional, Iterable, Any
from concurrent.futures import ProcessPoolExecutor, as_completed

import nltk
from tqdm import tqdm

from denserr.dataset._base import LargeCorpusSequentialDict
from denserr.model.load_model import LoadRetriever


def truncate_by_senteces(corpus_iter, offset, tokenizer, output_path) -> None:
    pass


def parallel_truncate_by_senteces(
    corpus: LargeCorpusSequentialDict,
    tokenizer: Any,
    output_path: Path,
    num_worker: int = 10,
):
    corpus_len = corpus.total
    data_size_per_worker = math.ceil(corpus_len / num_worker)
    max_workers = num_worker

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i, offset in enumerate(range(0, corpus_len, data_size_per_worker)):
            output_path = ""
            future = executor.submit(
                truncate_by_senteces, corpus, offset, tokenizer, output_path
            )
            futures.append(future)


def truncate_by_sentences(self, corpus: LargeCorpusSequentialDict) -> Path:
    retriever = LoadRetriever(self.dataset_name, self.model_name).load_retriever()
    tokenizer = retriever.tokenizer

    self.separator
    output_path = self.cache_csv_path()
    with output_path.open("w", newline="") as f:
        fieldnames = ["id", "text"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, doc in enumerate(tqdm(corpus, total=corpus.total)):
            doc_id = doc["id"]
            if len(doc["text"]) == 0:
                continue

            sentences = nltk.sent_tokenize(doc["text"])
            if len(sentences) <= 0:
                continue

            tokenized = tokenizer(sentences, add_special_tokens=False)

            accepted_ids = []
            accepted_ids_sum = 0
            for ids in tokenized["input_ids"]:
                if accepted_ids_sum + len(ids) < self.max_doc_len:
                    accepted_ids.append(ids)
                    accepted_ids_sum += len(ids)
                else:
                    break

            restored_sentences = tokenizer.batch_decode(
                accepted_ids, skip_special_tokens=True
            )
            truncated_text = " ".join(restored_sentences)

            new_doc = {
                "id": doc_id,
                "text": json.dumps(truncated_text, ensure_ascii=True),
            }
            writer.writerow(new_doc)

def index_with_multiprocessing(
    docs_iter, encoder_loader, batch_size, device, order: int
):
    print(f"index_with_multiprocessing of order {order}, {device}")
    docs = list(docs_iter)
    encoder = encoder_loader(device)
    passage_embedding = encoder.encode_corpus(docs, batch_size, convert_to_tensor=False)
    return (order, passage_embedding)
