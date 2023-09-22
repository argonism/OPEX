import gokart
import luigi
import numpy as np

import denserr
from denserr.model.load_model import LoadRetriever
from denserr.model.deepct_sent_parallel import PreprocessDeepCTSentParallel
from denserr.retrieve import (
    Rerank,
    Retrieve,
    RetrieveForDebug,
    Evaluate,
    EvaluatePerQuery,
    TuningBM25,
    TuningBM25WithMSMARCODoc,
)

# from denserr.analyzer.embedding_analyzer import (
#     EmbeddingAnalyze,
#     PerturbedEmbeddingAnalyze,
# )
# from denserr.analyzer.perturbation_text import PerturbationTextAnalyze
from denserr.analyzer.damaged_analyzer import (
    DamagedAnalyze,
    AnalyzeDamagedDistribution,
    ShowDamagedCases,
    DamagedEvaluate,
)
from denserr.analyzer.sentence_intact_analyzer import (
    SentenceInstactAnalyze,
    AnalyzeSentenceInstactDistribution,
    ShowSentIntactCases,
    CalcSentIntactStats,
)
from denserr.debug.debug_single_scoring import (
    DebugSingleScoring,
    DebugBatchSingleScoring,
)

from denserr.train.generate_dataset import GeneratePerturbedDataset

# from denserr.analyzer.perturbation_embedding import PerturbationEmbeddingAnalyze
# from denserr.analyzer.intact_analyzer import IntactAnalyze

if __name__ == "__main__":
    gokart.add_config("./conf/param.ini")
    gokart.run()
