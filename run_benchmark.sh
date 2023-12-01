
ZERO_STAGE=${1:-"2"}
SEQ_LENGTH=${2:-2048}
GLOBAL_BATCH_SIZE=${3:-32}
BATCH_SIZE_PER_GPU=$(( $GLOBAL_BATCH_SIZE / 8 ))

DS_CONFIG="./ds_config.json"
cat <<EOT > $DS_CONFIG
{
    "fp16": {
        "enabled": true,
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 8,
        "hysteresis": 2,
        "min_loss_scale": 1
    },
    "optimizer": {
        "type": "Adam"
    },
    "zero_optimization": {
        "stage": $ZERO_STAGE,
        "allgather_partitions": true,
        "allgather_bucket_size": 1e8,
        "overlap_comm": true,
        "reduce_scatter": true,
        "reduce_bucket_size": 1e8,
        "contiguous_gradients": true
    },

    "gradient_accumulation_steps": 1,
    "train_batch_size": $GLOBAL_BATCH_SIZE,
    "train_micro_batch_size_per_gpu":$BATCH_SIZE_PER_GPU,
    "zero_allow_untested_optimizer": true
}
EOT

mkdir -p ./logs

deepspeed benchmark.py \
	--grad_checkpoint \
	--deepspeed \
	--deepspeed_config $DS_CONFIG \
	--max_length $SEQ_LENGTH \
	2>&1 | tee -a ./logs/deepspeed_zero${ZERO_STAGE}_bs${GLOBAL_BATCH_SIZE}_seq${SEQ_LENGTH}.log