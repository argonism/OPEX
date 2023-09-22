torchrun --nproc_per_node=4 train_colbert.py \
  --output_dir trained/colbert/perturbed_msmarco-pas_b16_p2_180 \
  --model_name_or_path bert-base-uncased \
  --save_steps 500 \
  --dataset_name json \
  --train_dir resources/denserr/audmented/msmarco-passage/ance_add_random.jsonl \
  --fp16 \
  --per_device_train_batch_size 16 \
  --positive_passage_no_shuffle \
  --train_n_passages 2 \
  --learning_rate 1e-5 \
  --q_max_len 32 \
  --p_max_len 180 \
  --num_train_epochs 1 \
  --logging_steps 1 \
  --negatives_x_device \
  --overwrite_output_dir