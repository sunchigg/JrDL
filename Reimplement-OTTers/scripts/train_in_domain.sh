# bash train_in_domain.sh
export DATA_TYPE=in_domain
export ROOT_PATH=..
export DEVICE=1
export PRE_TRAINED={gpt2-small}
CUDA_VISIBLE_DEVICES=${DEVICE} \
python3 main.py \
--train_data_file ${ROOT_PATH}/data/${DATA_TYPE}/train \
--dev_data_file ${ROOT_PATH}/data/${DATA_TYPE}/dev \
--test_data_file ${ROOT_PATH}/data/${DATA_TYPE}/test \
--graph_path 2hops_100_directed_triple_filter.json \
--output_dir ${ROOT_PATH}/models/${DATA_TYPE}/grf-${DATA_TYPE} \
--source_length 32 \
--target_length 32 \
--model_type gpt2 \
--model_name_or_path ${ROOT_PATH}/models/gpt2-small \
--do_train \
--validate_steps -1 \
--per_gpu_train_batch_size 16 \
--per_gpu_eval_batch_size 16 \
--workers 7 \
--seed 42 \
--evaluate_metrics bleu \
--overwrite_output_dir \
--num_train_epochs 10 \
--learning_rate 1e-5 \
--aggregate_method max \
--alpha 3 \
--beta 5 \
--gamma 0.5 \
--weight_decay 0.01 \
--warmup_ratio 0.1 \
--logging_steps 20 \