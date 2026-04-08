#!/bin/bash
set -x
export VLLM_USE_FLASHINFER=0
export VLLM_ATTENTION_BACKEND=XFORMERS
export HF_ENDPOINT=https://hf-mirror.com
export HOME="/data/home/Yichen/CC-GRPO/eval_llm_safety"

export CUDA_VISIBLE_DEVICES=0

#/data/home/Yichen/CC-GRPO/verl/checkpoints/cc-grpo/lam-7B-05/global_step_100/hg_model
MODEL_NAME="saver-grpo-7B-new"
MODEL_PATH="/data/home/Yichen/CC-GRPO/verl/checkpoints/saver-grpo-7B/pure_grpo"
EVALUATOR_MODEL="/data/home/Yichen/data1/models/meta-llama/Llama-Guard-3-8B"

RUN_DATASETS="['wildchat']"
use_template=True

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
    datasets.wildjailbreak.use_template=$use_template \
    datasets.strongreject.use_template=$use_template \
    datasets.xstest.use_template=False \
    datasets.wildchat.use_template=$use_template \
    datasets.bsa.use_template=True \
    datasets.wildjailbreak.limit_num=100 \
    datasets.strongreject.limit_num=100 \
    datasets.wildchat.limit_num=1 \
    datasets.bsa.limit_num=-1 \
    datasets.wildjailbreak.eval.evaluator_model=$EVALUATOR_MODEL \
    2>&1 | tee $EXPERIMENT_LOG_DIR/eval.log