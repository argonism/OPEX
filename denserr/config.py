import luigi


class DenseErrConfig(luigi.Config):
    dataset_name = luigi.Parameter()
    model_name = luigi.Parameter()
    is_debug = luigi.BoolParameter()
    with_pyserini = luigi.BoolParameter()
    perturb_percent = luigi.IntParameter()
    sample_repeat_times = luigi.IntParameter(100)
    target_doc_rank = luigi.IntParameter(1)
    analyze_at_k = luigi.IntParameter()

    max_doc_len = luigi.IntParameter()
    use_pyterrier = luigi.BoolParameter()

    damaged_start_at = luigi.IntParameter()
    damaged_until = luigi.IntParameter()

    perturb_context = luigi.Parameter()
    perturb_position = luigi.Parameter("random")

    intact_start_at = luigi.IntParameter()
    intact_until = luigi.IntParameter()
