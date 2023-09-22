# Over-penalization against Extra Information in Neural IR Models

## Requirements
For experiments with robust04, follow the instructions on the following ir-datasets instructions to set up TREC disks 4 and 5.
https://ir-datasets.com/disks45.html#disks45/nocr/trec-robust-2004 

## Getting started

Install dependencies.

```python
conda env create -f=envs/denserr.yml
conda activate denserr
```

Run Sentence Deletion Analysis experiments

```shell
python main.py denserr.DamagedAnalyze --local-scheduler
```

Task result is output to `resources/denserr/analyzer/damaged_analyzer/{cache_file_name}`.

Then, correct and visualize ranking shift results

```shell
python scripts/compare_ranking_shifts.py \
resources/denserr/analyzer/damaged_analyzer/{cache_file_name}
```

You can compare multiple results with this script

```shell
python scripts/compare_ranking_shifts.py \
resources/denserr/analyzer/damaged_analyzer/{BM25_result_filename}
resources/denserr/analyzer/damaged_analyzer/{ANCE_result_filename}
resources/denserr/analyzer/damaged_analyzer/{ColBERT_result_filename}
resources/denserr/analyzer/damaged_analyzer/{DeepCT_result_filename}
resources/denserr/analyzer/damaged_analyzer/{SPLADE_result_filename}
```

To run Sentence Addition Analysis experiments, execute SentenceInstactAnalyze

```shell
python main.py denserr.SentenceInstactAnalyze --local-scheduler
```

When changing the datasets, models, and various settings used in the experiments, please edit `conf/param.ini`.

### For ColBERT, SPLADE

To resolve dependency issues, we have prepared an conda environment yml for both ColBERT and SPLADE. 
If you want to use either of these models, create and activate their respective conda envs.
If you are using pyenv, don't forget to set the appropriate Python version using pyenv local [version].


colbert: envs/colbert.yml
splade: envs/ptsplade.yml

