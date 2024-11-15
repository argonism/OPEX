torchrun --nproc_per_node=4 train_splade.py \
  --output_dir trained/splade/perturbed_msmarco-pas_b16_p2 \
  --model_name_or_path naver/splade-cocondenser-ensembledistil \
  --save_steps 500 \
  --dataset_name json \
  --train_dir resources/denserr/audmented/msmarco-passage/ance_add_random.jsonl \
  --fp16 \
  --per_device_train_batch_size 8 \
  --positive_passage_no_shuffle \
  --train_n_passages 2 \
  --learning_rate 1e-5 \
  --q_max_len 32 \
  --p_max_len 256 \
  --num_train_epochs 1 \
  --logging_steps 1 \
  --negatives_x_device \
  --overwrite_output_dir