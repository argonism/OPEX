from logging import getLogger
from typing import Dict
from pathlib import Path

from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRE

from denserr.model._base import DenseRetriever, Retriever
from denserr.model.ance import AnceTextEncoder, load_ance
from denserr.model.bm25 import BM25Retriever
from denserr.model.dpr import DPRTextEncoder
from denserr.model.pyterrier_dense import PyTDenseRetriever
from denserr.model.pyterrier_dense_parallel import PyTParallelDenseRetriever
from denserr.model.pyterrier_dense_parallel_sent import PyTParallelDenseSentRetriever

from denserr.utils.util import project_dir

logger = getLogger(__name__)


class LoadRetriever:
    def __init__(self, dataset_name: str, model_name: str) -> None:
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.batch_size = 16

    def load_ance(self, use_pyterrier: bool = True, debug: bool = False) -> Retriever:
        encoder = AnceTextEncoder()
        if use_pyterrier:
            retriever = PyTDenseRetriever(
                encoder,
                self.dataset_name,
                self.model_name,
                index_prefix="debug" if debug else "",
                topk=10000,
                segment_size=500_000,
            )
            return retriever
        else:
            model = DRE(encoder, batch_size=self.batch_size)
            retriever = EvaluateRetrieval(model, score_function="dot", k_values=[10000])
            # prefix = self.dataset_name
            # ext = "flat"
            wrapped_retriever = DenseRetriever(retriever)
            return wrapped_retriever

    def __load_ance_parallel(
        self, model_confs: Dict, debug: bool, topk: int = 10000
    ) -> Retriever:
        retriever = PyTParallelDenseRetriever(
            load_ance,
            self.dataset_name,
            self.model_name,
            model_confs=model_confs,
            index_prefix="debug" if debug else "",
            topk=topk,
            segment_size=1_000_000,
            overwrite=True,
        )
        return retriever

    def load_ance_parallel(self, debug: bool = False, topk: int = 10000) -> Retriever:
        return self.__load_ance_parallel({}, debug=debug, topk=topk)

    def __load_ance_sent(self, window_size: int, debug: bool = False) -> Retriever:
        retriever = PyTParallelDenseSentRetriever(
            load_ance,
            self.dataset_name,
            self.model_name,
            index_prefix="debug" if debug else "",
            topk=10000,
            segment_size=1_000_000,
            window_size=window_size,
        )
        return retriever

    def load_ance_sent(self, debug: bool = False) -> Retriever:
        return self.__load_ance_sent(1, debug=debug)

    def load_ance_sent_w2(self, debug: bool = False) -> Retriever:
        return self.__load_ance_sent(2, debug=debug)

    def load_ance_sent_w3(self, debug: bool = False) -> Retriever:
        return self.__load_ance_sent(3, debug=debug)

    def load_ance_sent_w4(self, debug: bool = False) -> Retriever:
        return self.__load_ance_sent(4, debug=debug)

    def load_ance_sent_w5(self, debug: bool = False) -> Retriever:
        return self.__load_ance_sent(5, debug=debug)

    def load_ance_sent_w10(self, debug: bool = False) -> Retriever:
        return self.__load_ance_sent(10, debug=debug)

    def load_ance_sent_w15(self, debug: bool = False) -> Retriever:
        return self.__load_ance_sent(15, debug=debug)

    def load_ance_sent_w20(self, debug: bool = False) -> Retriever:
        return self.__load_ance_sent(20, debug=debug)

    def load_ance_sent_w25(self, debug: bool = False) -> Retriever:
        return self.__load_ance_sent(25, debug=debug)

    def load_ance_random(
        self, use_pyterrier: bool = True, debug: bool = False
    ) -> Retriever:
        model_path = project_dir.joinpath("trained/perturbed_msmarco-pas_random_24")
        encoder = AnceTextEncoder(
            str(model_path), tokenizer_name="trained/perturbed_msmarco-pas_random_24"
        )
        retriever = PyTDenseRetriever(
            encoder,
            self.dataset_name,
            self.model_name,
            index_prefix="debug" if debug else "",
            topk=10000,
            segment_size=1_000_000,
        )
        return retriever

    def load_ance_random_500(
        self, use_pyterrier: bool = True, debug: bool = False
    ) -> Retriever:
        model_path = project_dir.joinpath("trained/perturbed_msmarco-pas_random_24")
        model_confs = {
            "model_path": str(model_path),
            "tokenizer_name": model_path,
        }
        return self.__load_ance_parallel(model_confs, debug=debug)

    def load_ance_random_500_cont_w_pert(
        self, use_pyterrier: bool = True, debug: bool = False
    ) -> Retriever:
        # model_path = project_dir.joinpath("trained/perturbed_msmarco-pas_random_24")
        model_confs = {
            "model_path": str(model_path),
            "tokenizer_name": "castorini/ance-msmarco-passage",
        }
        return self.__load_ance_parallel(model_confs, debug=debug)

    def load_ance_random_mse(
        self, use_pyterrier: bool = True, debug: bool = False
    ) -> Retriever:
        model_path = project_dir.joinpath("trained/perturbed_msmarco-pas_random_24_mse")
        model_confs = {
            "model_path": str(model_path),
            "tokenizer_name": "castorini/ance-msmarco-passage",
        }
        return self.__load_ance_parallel(model_confs, debug=debug)

    def load_ance_random_mse_500(
        self, use_pyterrier: bool = True, debug: bool = False
    ) -> Retriever:
        model_path = project_dir.joinpath(
            "trained/perturbed_msmarco-pas_random_24_mse/checkpoint-500"
        )
        model_confs = {
            "model_path": str(model_path),
            "tokenizer_name": "castorini/ance-msmarco-passage",
        }
        return self.__load_ance_parallel(model_confs, debug=debug)

    def load_ance_random_mse_if_less_500(
        self, use_pyterrier: bool = True, debug: bool = False
    ) -> Retriever:
        model_path = project_dir.joinpath(
            "trained/perturbed_msmarco-pas_random_24_margin/contrastive_with_perturbed_mse_if_less/checkpoint-500"
        )
        model_confs = {
            "model_path": str(model_path),
            "tokenizer_name": "castorini/ance-msmarco-passage",
        }
        return self.__load_ance_parallel(model_confs, debug=debug)

    def load_ance_random_mse_if_less_2000(
        self, use_pyterrier: bool = True, debug: bool = False
    ) -> Retriever:
        model_path = project_dir.joinpath(
            "trained/perturbed_msmarco-pas_random_24_margin/contrastive_with_perturbed_mse_if_less/checkpoint-2000"
        )
        model_confs = {
            "model_path": str(model_path),
            "tokenizer_name": "castorini/ance-msmarco-passage",
        }
        return self.__load_ance_parallel(model_confs, debug=debug)

    def load_ance_random_wo_pert(
        self, use_pyterrier: bool = True, debug: bool = False
    ) -> Retriever:
        model_path = project_dir.joinpath(
            "trained/perturbed_msmarco-pas_random_24_wo_perturbedloss"
        )
        model_confs = {
            "model_path": str(model_path),
        }
        return self.__load_ance_parallel(model_confs, debug=debug)

    def load_ance_random_1epoch(
        self, use_pyterrier: bool = True, debug: bool = False
    ) -> Retriever:
        model_path = project_dir.joinpath("trained/perturbed_msmarco-pas_random_24")
        model_confs = {
            "model_path": str(model_path),
            "tokenizer_name": str(model_path),
        }
        return self.__load_ance_parallel(model_confs, debug=debug)

    def load_ance_random_wo_pert_500(
        self, use_pyterrier: bool = True, debug: bool = False
    ) -> Retriever:
        model_path = project_dir.joinpath(
            "trained/perturbed_msmarco-pas_random_24_wo_perturbloss/checkpoint-500"
        )
        model_confs = {
            "model_path": str(model_path),
            "tokenizer_name": "castorini/ance-msmarco-passage",
        }
        return self.__load_ance_parallel(model_confs, debug=debug)

    def load_dpr(self, use_pyterrier: bool = True, debug: bool = False) -> Retriever:
        encoder = DPRTextEncoder()
        retriever = PyTDenseRetriever(
            encoder,
            self.dataset_name,
            self.model_name,
            index_prefix="debug" if debug else "",
            topk=10000,
            segment_size=500_000,
        )
        return retriever

    def load_bm25(self, debug: bool = False) -> BM25Retriever:
        retriever = BM25Retriever.get_instance()
        retriever.set_params(
            self.dataset_name,
            self.model_name,
            topk=10000,
            index_prefix="debug" if debug else "",
            config={"bm25.b": 0.2, "bm25.k_1": 0.9},
        )
        return retriever

    def load_bm25_robust04(self, debug: bool = False) -> BM25Retriever:
        retriever = BM25Retriever.get_instance()
        retriever.set_params(
            self.dataset_name,
            self.model_name,
            topk=10000,
            index_prefix="debug" if debug else "",
            config={"bm25.b": 0.2, "bm25.k_1": 0.9},
        )
        return retriever

    def load_bm25_msmarco_doc(self, debug: bool = False) -> BM25Retriever:
        retriever = BM25Retriever.get_instance()
        retriever.set_params(
            self.dataset_name,
            self.model_name,
            topk=10000,
            index_prefix="debug" if debug else "",
            config={"bm25.b": 0.8, "bm25.k_1": 1.2},
        )
        return retriever

    def load_deepct(self, debug: bool = False) -> Retriever:
        from denserr.model.deepct import DeepctRetriever

        retriever = DeepctRetriever.get_instance()
        retriever.set_params(
            self.dataset_name, topk=10000, index_prefix="debug" if debug else ""
        )
        return retriever

    def load_deepct_sent(self) -> Retriever:
        from denserr.model.deepct_sent import DeepctSentRetriever

        retriever = DeepctSentRetriever.get_instance()
        retriever.set_params(self.dataset_name, topk=10000)
        return retriever

    def _load_deepct_sent_parallel(
        self, debug: bool = False, window_size: int = 1
    ) -> Retriever:
        try:
            from denserr.model.deepct_sent_parallel import (
                DeepctSentParallelRetriever,
            )
        except ModuleNotFoundError as e:
            logger.error("Failed to import DeepctSentParallelRetriever ")
            raise e
        retriever = DeepctSentParallelRetriever.get_instance()
        retriever.set_params(
            self.dataset_name,
            topk=10000,
            index_prefix="debug" if debug else "",
            window_size=window_size,
        )
        return retriever

    def load_colbert(self, debug: bool = False) -> Retriever:
        from denserr.model.colbert import ColbertRetriever

        model_path = "http://www.dcs.gla.ac.uk/~craigm/ecir2021-tutorial/colbert_model_checkpoint.zip"
        dataset_name = self.dataset_name
        model_name = self.model_name
        return ColbertRetriever(
            model_path, dataset_name, model_name, index_prefix="debug" if debug else ""
        )

    def load_colbert_sent(self, debug: bool = False) -> Retriever:
        from denserr.model.colbert_sent import ColbertSentRetriever

        model_path = "http://www.dcs.gla.ac.uk/~craigm/ecir2021-tutorial/colbert_model_checkpoint.zip"
        dataset_name = self.dataset_name
        model_name = self.model_name
        return ColbertSentRetriever(
            model_path, dataset_name, model_name, index_prefix="debug" if debug else ""
        )

    def load_colbert_sent_w2(self, debug: bool = False) -> Retriever:
        from denserr.model.colbert_sent import ColbertSentRetriever
        import pyterrier as pt

        if not pt.started():
            pt.init()

        model_path = "http://www.dcs.gla.ac.uk/~craigm/ecir2021-tutorial/colbert_model_checkpoint.zip"
        dataset_name = self.dataset_name
        model_name = self.model_name
        return ColbertSentRetriever(
            model_path,
            dataset_name,
            model_name,
            index_prefix="debug" if debug else "",
            window_size=2,
        )

    def load_ptsplade(self, debug: bool = False) -> Retriever:
        try:
            import pyterrier as pt

            if not pt.started():
                pt.init()
            from denserr.model.pt_splade_model import PtSpladeRetriever
        except ModuleNotFoundError as e:
            logger.error("Failed to import PtSpladeRetriever ")
            logger.error(
                "It seems that splade env does not activated or python used is not for ptsplade env"
            )
            raise e
        retriever = PtSpladeRetriever.get_instance()
        retriever.set_params(
            self.dataset_name, topk=10000, index_prefix="debug" if debug else ""
        )
        return retriever

    def load_ptsplade_sent(self, debug: bool = False) -> Retriever:
        try:
            import pyterrier as pt

            if not pt.started():
                pt.init()
            from denserr.model.pt_splade_sent import PtSpladeSentRetriever
        except ModuleNotFoundError as e:
            logger.error("Failed to import PtSpladeRetriever ")
            logger.error(
                "It seems that splade env does not activated or python used is not for ptsplade env"
            )
            raise e
        retriever = PtSpladeSentRetriever.get_instance()
        retriever.set_params(
            self.dataset_name, topk=10000, index_prefix="debug" if debug else ""
        )
        return retriever

    def _load_ptsplade_parallel(
        self, model_path: str, debug: bool = False
    ) -> Retriever:
        try:
            from denserr.model.pt_splade_model_parallel import (
                PtSpladeParallelRetriever,
            )
        except ModuleNotFoundError as e:
            logger.error("Failed to import PtSpladeRetriever ")
            logger.error(
                "It seems that splade env does not activated or python used is not for ptsplade env"
            )
            raise e
        retriever = PtSpladeParallelRetriever.get_instance()
        retriever.set_params(
            self.dataset_name,
            self.model_name,
            model_path,
            topk=10000,
            index_prefix="debug" if debug else "",
        )
        return retriever

    def load_ptsplade_parallel(self, debug: bool = False) -> Retriever:
        return self._load_ptsplade_parallel(
            "naver/splade-cocondenser-ensembledistil", debug=debug
        )

    def load_ptsplade_parallel_500(self, debug: bool = False) -> Retriever:
        return self._load_ptsplade_parallel(
            str(
                project_dir.joinpath(
                    "trained/splade/perturbed_msmarco-pas_b16_p2/checkpoint-500"
                )
            ),
            debug=debug,
        )

    def load_ptsplade_parallel_10000(self, debug: bool = False) -> Retriever:
        return self._load_ptsplade_parallel(
            str(
                project_dir.joinpath(
                    "trained/splade/perturbed_msmarco-pas_b16_p2/checkpoint-10000"
                )
            ),
            debug=debug,
        )

    def _load_ptsplade_sent_windowed_parallel(
        self, window_size: int, debug: bool = False
    ) -> Retriever:
        try:
            import pyterrier as pt

            if not pt.started():
                pt.init()
            from denserr.model.pt_splade_sent_parallel import (
                PtSpladeSentParallelRetriever,
            )
        except ModuleNotFoundError as e:
            logger.error("Failed to import PtSpladeRetriever ")
            logger.error(
                "It seems that splade env does not activated or python used is not for ptsplade env"
            )
            raise e
        retriever = PtSpladeSentParallelRetriever.get_instance()
        retriever.set_params(
            self.dataset_name,
            topk=10000,
            index_prefix="debug" if debug else "",
            window_size=window_size,
        )
        return retriever

    def load_ptsplade_sent_parallel(self, debug: bool = False) -> Retriever:
        return self._load_ptsplade_sent_windowed_parallel(1, debug=debug)

    def load_ptsplade_sent_w2_parallel(self, debug: bool = False) -> Retriever:
        return self._load_ptsplade_sent_windowed_parallel(2, debug=debug)

    def load_ptsplade_sent_w3_parallel(self, debug: bool = False) -> Retriever:
        return self._load_ptsplade_sent_windowed_parallel(3, debug=debug)

    def load_ptsplade_sent_w5_parallel(self, debug: bool = False) -> Retriever:
        return self._load_ptsplade_sent_windowed_parallel(5, debug=debug)

    def load_ptsplade_sent_w6_parallel(self, debug: bool = False) -> Retriever:
        return self._load_ptsplade_sent_windowed_parallel(6, debug=debug)

    def load_ptsplade_sent_w9_parallel(self, debug: bool = False) -> Retriever:
        return self._load_ptsplade_sent_windowed_parallel(9, debug=debug)

    def load_ptsplade_sent_w10_parallel(self, debug: bool = False) -> Retriever:
        return self._load_ptsplade_sent_windowed_parallel(10, debug=debug)

    def load_ptsplade_sent_w12_parallel(self, debug: bool = False) -> Retriever:
        return self._load_ptsplade_sent_windowed_parallel(12, debug=debug)

    def load_ptsplade_sent_w15_parallel(self, debug: bool = False) -> Retriever:
        return self._load_ptsplade_sent_windowed_parallel(15, debug=debug)

    def load_ptsplade_sent_w20_parallel(self, debug: bool = False) -> Retriever:
        return self._load_ptsplade_sent_windowed_parallel(20, debug=debug)

    def load_ptsplade_sent_w25_parallel(self, debug: bool = False) -> Retriever:
        return self._load_ptsplade_sent_windowed_parallel(25, debug=debug)

    def load_retriever(
        self,
        use_pyterrier: bool = True,
        debug: bool = False,
        topk: int = 10000,
    ) -> Retriever:
        if self.model_name == "ance":
            retriever: Retriever = self.load_ance(
                use_pyterrier=use_pyterrier, debug=debug
            )
        elif self.model_name == "ance-parallel":
            retriever = self.load_ance_parallel(debug=debug, topk=topk)
        elif self.model_name == "ance-sent":
            retriever = self.load_ance_sent(debug=debug)
        elif self.model_name == "ance-sent-w2":
            retriever = self.load_ance_sent_w2(debug=debug)
        elif self.model_name == "ance-sent-w3":
            retriever = self.load_ance_sent_w3(debug=debug)
        elif self.model_name == "ance-sent-w4":
            retriever = self.load_ance_sent_w4(debug=debug)
        elif self.model_name == "ance-sent-w5":
            retriever = self.load_ance_sent_w5(debug=debug)
        elif self.model_name == "ance-sent-w10":
            retriever = self.load_ance_sent_w10(debug=debug)
        elif self.model_name == "ance-sent-w15":
            retriever = self.load_ance_sent_w15(debug=debug)
        elif self.model_name == "ance-sent-w20":
            retriever = self.load_ance_sent_w20(debug=debug)
        elif self.model_name == "ance-sent-w25":
            retriever = self.load_ance_sent_w25(debug=debug)
        elif self.model_name == "ance-random":
            retriever = self.load_ance_random(debug=debug)
        elif self.model_name == "ance-random-1epoch":
            retriever = self.load_ance_random_1epoch(debug=debug)
        elif self.model_name == "ance-random-wo-pert":
            retriever = self.load_ance_random_wo_pert(debug=debug)
        elif self.model_name == "ance-random-5500":
            retriever = self.load_ance_random_5500(debug=debug)
        elif self.model_name == "ance-random-500":
            retriever = self.load_ance_random_500(debug=debug)
        elif self.model_name == "ance-random-500-cont-w-pert":
            retriever = self.load_ance_random_500_cont_w_pert(debug=debug)
        elif self.model_name == "ance-random-mse":
            retriever = self.load_ance_random_mse(debug=debug)
        elif self.model_name == "ance-random-mse-500":
            retriever = self.load_ance_random_mse_500(debug=debug)
        elif self.model_name == "ance-random-mse-if-less-500":
            retriever = self.load_ance_random_mse_if_less_500(debug=debug)
        elif self.model_name == "ance-random-mse-if-less-2000":
            retriever = self.load_ance_random_mse_if_less_2000(debug=debug)
        elif self.model_name == "ance-random-wo-part-500":
            retriever = self.load_ance_random_wo_pert_500(debug=debug)
        elif self.model_name == "dpr":
            retriever = self.load_dpr(use_pyterrier=use_pyterrier, debug=debug)
        elif self.model_name == "colbert":
            retriever = self.load_colbert(debug=debug)
        elif self.model_name == "colbert-sent":
            retriever = self.load_colbert_sent(debug=debug)
        elif self.model_name == "colbert-sent-w2":
            retriever = self.load_colbert_sent_w2(
                debug=debug,
            )
        elif self.model_name == "bm25":
            retriever = self.load_bm25(debug=debug)
        elif self.model_name == "bm25-robust04":
            retriever = self.load_bm25_robust04(debug=debug)
        elif self.model_name == "bm25-msmarco-doc":
            retriever = self.load_bm25_msmarco_doc(debug=debug)
        elif self.model_name == "deepct":
            retriever = self.load_deepct(debug=debug)
        elif self.model_name == "deepct-sent":
            retriever = self.load_deepct_sent()
        elif self.model_name == "deepct-sent-parallel":
            retriever = self._load_deepct_sent_parallel(debug=debug, window_size=1)
        elif self.model_name == "splade":
            retriever = self.load_splade()
        elif self.model_name == "ptsplade":
            retriever = self.load_ptsplade(debug=debug)
        elif self.model_name == "ptsplade-parallel":
            retriever = self.load_ptsplade_parallel(debug=debug)
        elif self.model_name == "ptsplade-parallel-500":
            retriever = self.load_ptsplade_parallel_500(debug=debug)
        elif self.model_name == "ptsplade-parallel-10000":
            retriever = self.load_ptsplade_parallel_10000(debug=debug)
        elif self.model_name == "ptsplade-sent":
            retriever = self.load_ptsplade_sent(debug=debug)
        elif self.model_name == "ptsplade-sent-parallel":
            retriever = self.load_ptsplade_sent_parallel(debug=debug)
        elif self.model_name == "ptsplade-sent-w2-parallel":
            retriever = self.load_ptsplade_sent_w2_parallel(debug=debug)
        elif self.model_name == "ptsplade-sent-w3-parallel":
            retriever = self.load_ptsplade_sent_w3_parallel(debug=debug)
        elif self.model_name == "ptsplade-sent-w5-parallel":
            retriever = self.load_ptsplade_sent_w5_parallel(debug=debug)
        elif self.model_name == "ptsplade-sent-w6-parallel":
            retriever = self.load_ptsplade_sent_w6_parallel(debug=debug)
        elif self.model_name == "ptsplade-sent-w9-parallel":
            retriever = self.load_ptsplade_sent_w9_parallel(debug=debug)
        elif self.model_name == "ptsplade-sent-w10-parallel":
            retriever = self.load_ptsplade_sent_w10_parallel(debug=debug)
        elif self.model_name == "ptsplade-sent-w12-parallel":
            retriever = self.load_ptsplade_sent_w12_parallel(debug=debug)
        elif self.model_name == "ptsplade-sent-w15-parallel":
            retriever = self.load_ptsplade_sent_w15_parallel(debug=debug)
        elif self.model_name == "ptsplade-sent-w20-parallel":
            retriever = self.load_ptsplade_sent_w20_parallel(debug=debug)
        elif self.model_name == "ptsplade-sent-w25-parallel":
            retriever = self.load_ptsplade_sent_w25_parallel(debug=debug)
        else:
            raise ValueError(f"Unknown model: {self.model_name}")
        return retriever
