
ds_config=${1:-"configs/zero2.json"}

deepspeed benchmark.py \
	--grad_checkpoint \
	--deepspeed \
	--deepspeed_config $ds_config