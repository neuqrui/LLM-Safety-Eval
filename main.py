import os
import gc
import argparse
import sys
import torch
from omegaconf import OmegaConf

# vLLM 与 Transformers 依赖
try:
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
except ImportError:
    print("⚠️ 警告: 未找到 vllm 或 transformers，仅限 API 模式可用。", file=sys.stderr)
    LLM, SamplingParams, AutoTokenizer = None, None, None

# 导入你本地的工具包 (请确保它们在 utils 目录下)
from utils.data_handler import load_and_prep_data, process_and_save_results
from utils.eval_engine import run_eval_bsa, run_eval_strongreject, run_eval_xstest, run_eval_guard_vllm, \
    append_to_summary_file, run_eval_frr


def get_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Unified Batch Inference & Eval Pipeline")
    parser.add_argument("--config", type=str, default="config.yaml", help="基础 config.yaml 路径")
    # parse_known_args 会把 --config 之外的参数（即 key=value 覆盖项）收集到 unknown 列表中
    args, unknown = parser.parse_known_args()
    return args, unknown


def setup_directories(ds_name):
    """统一创建标准输出目录体系"""
    paths = {
        "infer": f"outputs/infer/{ds_name}",
        "eval": f"outputs/eval/{ds_name}",
        "result": "outputs/result"
    }
    for p in paths.values():
        os.makedirs(p, exist_ok=True)
    return paths


def apply_template(prompts, config, tokenizer):
    """将 Prompt 包装为目标模型支持的 Chat Template"""
    if not config.get('use_template', False) or tokenizer is None:
        return prompts

    sys_template = config.get('sys_template', "")
    formatted_prompts = []

    if hasattr(tokenizer, 'apply_chat_template') and config.get('use_tokenizer_template'):
        for prompt in prompts:
            messages = []
            if sys_template:
                messages.append({"role": "system", "content": sys_template})
            messages.append({"role": "user", "content": prompt})
            formatted_prompts.append(
                tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            )
    else:
        formatted_prompts = prompts
    return formatted_prompts


def main():
    args, unknown_args = get_args()

    # ==========================================================================
    # 0. 配置解析与合并 (OmegaConf)
    # ==========================================================================
    if not os.path.exists(args.config):
        print(f"❌ 错误: 找不到配置文件 {args.config}", file=sys.stderr)
        sys.exit(1)

    # 加载基础 YAML
    base_config = OmegaConf.load(args.config)

    # 将命令行参数 (例如 model.name=Safe-sft-1.5B) 转换为 OmegaConf 对象
    cli_config = OmegaConf.from_cli(unknown_args)

    # 深度合并：命令行参数覆盖基础参数
    merged_config = OmegaConf.merge(base_config, cli_config)

    # 转换为原生的 Python 字典，方便下游的旧代码逻辑直接调用
    config = OmegaConf.to_container(merged_config, resolve=True)

    print("\n" + "=" * 50)
    print(" 🔧 最终生效配置 (Base YAML + CLI Overrides)")
    print("=" * 50)
    print(OmegaConf.to_yaml(merged_config))
    print("=" * 50 + "\n")

    model_config = config.get('model', {})
    run_datasets = config.get('run_datasets', [])
    api_config = config.get('api_config', {})

    if not run_datasets:
        print("❌ 错误: 未指定需要运行的数据集 (run_datasets 为空)！", file=sys.stderr)
        return

    print(f"🚀 初始化 Pipeline | 模型: {model_config.get('name')} | 模式: {model_config.get('mode')}")

    # ==========================================================================
    # 阶段 1: 集中数据准备
    # ==========================================================================
    print("\n" + "=" * 40 + "\n 📦 阶段 1: 数据准备 \n" + "=" * 40)
    task_payloads = {}

    if model_config.get('mode') == 'vllm' and LLM is not None:
        tokenizer = AutoTokenizer.from_pretrained(model_config['path'], trust_remote_code=True)
    else:
        tokenizer = None

    for ds_name in run_datasets:
        ds_config = config.get('datasets', {}).get(ds_name)
        if not ds_config:
            print(f"⚠️ 警告: 找不到数据集 [{ds_name}] 的配置，跳过。")
            continue

        print(f"正在加载数据集: {ds_name}...")
        orig_prompts, metadata = load_and_prep_data(ds_name, ds_config)
        final_prompts = apply_template(orig_prompts, ds_config, tokenizer)

        task_payloads[ds_name] = {
            "orig_prompts": orig_prompts,
            "final_prompts": final_prompts,
            "metadata": metadata,
            "config": ds_config
        }
        print(f"  └─ 成功准备了 {len(final_prompts)} 条数据。")

    if not task_payloads:
        print("❌ 错误: 没有成功准备任何数据，程序退出。", file=sys.stderr)
        return

    # ==========================================================================
    # 阶段 2: 集中推理
    # ==========================================================================
    print("\n" + "=" * 40 + "\n 🏃 阶段 2: 集中推理 \n" + "=" * 40)
    infer_results = {}

    if model_config.get('mode') == 'vllm':
        print(f"🔥 加载 vLLM 模型: {model_config['path']}")
        llm = LLM(
            model=model_config['path'],
            tensor_parallel_size=model_config.get('tensor_parallel_size', 1),
            trust_remote_code=True
        )

        for ds_name, payload in task_payloads.items():
            print(f"\n▶️ 开始推理 [{ds_name}]")
            ds_config = payload['config']
            sp_cfg = ds_config.get('sampling_params', {})
            n_gen = ds_config.get('dataset_specific_params', {}).get('pass_k', sp_cfg.get('n', 1))

            sampling_params = SamplingParams(
                n=n_gen,
                temperature=sp_cfg.get('temperature', 0.5),
                top_p=sp_cfg.get('top_p', 0.9),
                max_tokens=sp_cfg.get('max_tokens', 4096),
                stop_token_ids=[tokenizer.eos_token_id] if tokenizer else None
            )

            # 执行 vLLM 批量生成
            outputs = llm.generate(payload['final_prompts'], sampling_params)

            # 解析输出文本
            raw_outputs_nested = [[comp.text.strip() for comp in out.outputs] for out in outputs]

            # 保存推理结果
            dirs = setup_directories(ds_name)
            output_file = process_and_save_results(
                ds_name, model_config['name'], dirs['infer'], ds_config,
                payload['orig_prompts'], payload['final_prompts'], raw_outputs_nested, payload['metadata']
            )
            infer_results[ds_name] = output_file

        # ⚠️ 关键步骤：彻底释放主模型显存，以防后续 Llama-Guard 等评测模型 OOM
        print("\n🧹 推理全部结束，释放 vLLM 显存...")
        del llm
        gc.collect()
        torch.cuda.empty_cache()

    else:
        print("API 模式: 在此处对接你的 call_llm_parallel 逻辑...")
        # (如果您主要评测的是本地微调模型，vLLM分支已足够完成任务)

    # ==========================================================================
    # 阶段 3: 集中评测
    # ==========================================================================
    print("\n" + "=" * 40 + "\n ⚖️ 阶段 3: 集中评测 \n" + "=" * 40)

    for ds_name in run_datasets:
        if ds_name not in infer_results:
            continue

        print(f"\n📝 正在评测 [{ds_name}]...")
        dirs = setup_directories(ds_name)
        infer_path = infer_results[ds_name]
        ds_config = config['datasets'][ds_name]

        eval_res = {}
        eval_config = ds_config.get('eval', {})

        try:
            if ds_name == 'bsa':
                eval_res = run_eval_bsa(model_config['name'], infer_path, dirs['eval'], api_config, eval_config)

            elif ds_name == 'strongreject':
                eval_method = eval_config.get('eval_method', 'guard')
                if eval_method == 'guard':
                    eval_res = run_eval_guard_vllm(model_config['name'], infer_path, dirs['eval'], api_config,
                                                   eval_config)
                elif eval_method == 'rubric':
                    eval_res = run_eval_strongreject(model_config['name'], infer_path, dirs['eval'], api_config)

            elif ds_name == 'xstest':
                eval_res = run_eval_xstest(model_config['name'], infer_path, dirs['eval'], api_config)

            elif ds_name in ['wildchat', 'wildjailbreak']:
                eval_res = run_eval_guard_vllm(model_config['name'], infer_path, dirs['eval'], api_config, eval_config)

            elif ds_name in ['oktest', 'phtest', 'falsereject', 'xstest-or']:
                eval_res = run_eval_frr(model_config['name'], infer_path, dirs['eval'], api_config)
                
            # 写入统一的 Summary 日志
            summary_data = eval_res.get('summary_data')
            if summary_data is not None:
                ds_config = config.get('datasets', {}).get(ds_name, {})
                append_to_summary_file(model_config['name'], ds_name, dirs['result'], summary_data, ds_config=ds_config)
                print(f"  └─ 评测完成！Summary 已保存。")
            else:
                print(f"  └─ ⚠️ 评测完成，但未返回 Summary 数据。")

        except Exception as e:
            print(f"❌ 评测数据集 [{ds_name}] 时发生错误: {str(e)}", file=sys.stderr)

    experiment_log_dir = os.environ.get('EXPERIMENT_LOG_DIR', 'outputs/result')
    print(f"\n🎉 评测流水线全部运行完毕！")
    print(f"📁 最终摘要和图表均已保存在: {experiment_log_dir} 或 outputs/result/ 下。")


if __name__ == "__main__":
    main()