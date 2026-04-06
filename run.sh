#!/bin/bash
set -x
export VLLM_USE_FLASHINFER=0
export VLLM_ATTENTION_BACKEND=XFORMERS
export HF_ENDPOINT=https://hf-mirror.com
export HOME="/data/home/Yichen/CC-GRPO/eval_llm_safety"

export CUDA_VISIBLE_DEVICES=4


MODEL_NAME="saver-ppo-15B"
MODEL_PATH="/data/home/Yichen/CC-GRPO/verl/checkpoints/saver-ppo-15B"
EVALUATOR_MODEL="/data/home/Yichen/data1/models/meta-llama/Llama-Guard-3-8B"

RUN_DATASETS="['xstest', 'bsa', 'wildjailbreak', 'strongreject']"

CURRENT_TIME=$(date +"%Y%m%d_%H%M%S")
export EXPERIMENT_LOG_DIR="$HOME/experiment_logs/eval_${MODEL_NAME}_${CURRENT_TIME}"
mkdir -p $EXPERIMENT_LOG_DIR

python3 main.py \
    --config config.yaml \
    model.name=$MODEL_NAME \
    model.path=$MODEL_PATH \
    run_datasets="$RUN_DATASETS" \
    api_config.eval_api_model="gpt-4o-mini" \
    api_config.max_workers=32 \
    datasets.wildjailbreak.use_template=False \
    datasets.strongreject.use_template=False \
    datasets.xstest.use_template=False \
    datasets.bsa.use_template=True \
    datasets.wildjailbreak.limit_num=100 \
    datasets.strongreject.limit_num=100 \
    datasets.bsa.limit_num=-1 \
    datasets.wildjailbreak.eval.evaluator_model=$EVALUATOR_MODEL \
    2>&1 | tee $EXPERIMENT_LOG_DIR/eval.log