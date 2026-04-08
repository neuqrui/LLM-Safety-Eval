# 🛡️ LLM Safety Alignment Evaluation Pipeline

一个面向大语言模型安全评测的统一流水线，支持：
- vLLM 批量推理
- 多数据集自动加载与预处理
- API Judge 与 Llama-Guard 评测
- 统一汇总结果到 `outputs/result/<model_name>_summary.txt`

项目主流程在 `main.py` 中分为三阶段：数据准备 -> 推理 -> 评测。推理结束后会主动释放显存，再进入评测阶段，减少 OOM 风险。

- **🚀 推理与评测解耦**：一次性加载生成模型（vLLM）完成所有数据集的推理，随后强行释放显存，再集中加载评测模型（如 Llama-Guard）或调用 API，杜绝 OOM。
- **⚙️ 动态配置驱动 (OmegaConf)**：完全抛弃繁杂的 `argparse`，通过单一 `config.yaml` 管理全局。支持在命令行通过 `key.subkey=value` 动态覆盖配置。
- **📊 自动化图表与报告**：评测结束后自动在 `outputs/result/` 目录下生成易读的纯文本 Summary 对齐表格，以及必要的统计图表（如 XSTest 的条形图）。
- **🧠 兼容长推理模型**：原生支持包含 `<think>...</think>` 标签的 Reasoning 模型评测（如 DeepSeek-R1 系列）。

---

## 📚 目前支持的评测集

本框架已深度集成以下主流安全与对齐评测集：

1. **StrongReject**
   - **评测目标**：模型对高强度越狱 (Jailbreak) 攻击的防御能力。
   - **评测方式**：支持基于 API 的 `rubric` 多维打分，以及基于 `Llama-Guard` 的 ASR (攻击成功率) 评测。
2. **BSA (Beyond Safe Answer)**
   - **评测目标**：评估模型回复的绝对安全性，以及其内部推理过程（Thinking process）是否覆盖了所有潜在的风险点。
   - **评测方式**：API Judge (Safe_Ans_Check + Thk_Acc_Judge) -> 综合 F-Score 惩罚计算。
3. **XSTest**
   - **评测目标**：评估模型是否存在“过度拒绝” (Over-refusal) 现象（即拒绝回答实际上安全的 prompt）。
   - **评测方式**：API 文本分类 (Full Compliance / Full Refusal / Partial Refusal)。
4. **WildChat**
   - **评测目标**：基于真实世界用户交互的有害性/毒性测试。
   - **评测方式**：Llama-Guard 自动化判定。
5. **WildJailbreak**
   - **评测目标**：大规模对抗性越狱测试集。
   - **评测方式**：Llama-Guard 自动化判定。

---

## 📂 项目结构

```text
eval_llm_safety/
├── run.sh                  # 🚀 一键启动入口脚本
├── main.py                 # 🧠 核心调度引擎 (配置解析 -> 推理 -> 释放显存 -> 评测)
├── config.yaml             # 📄 统一的静态配置文件
├── utils/                  # 🛠️ 工具包目录
│   ├── __init__.py
│   ├── data_handler.py     # 负责各数据集的下载、预处理与格式化保存
│   ├── eval_engine.py      # 核心评测逻辑、指标计算与摘要生成
│   ├── call_llm.py         # 并发 API 请求封装工具
│   ├── prompts.py          # LLM-as-a-Judge 的 Prompt 模板库
│   └── evaluate.py         # BSA 专用指标计算工具
└── outputs/                # 📁 自动化输出目录
    ├── infe/               # 模型推理生成的原始回复 (JSON/CSV)
    ├── eval/               # Judge 模型的打分中间结果
    └── result/             # 最终的 Summary 报告与可视化图表
```

## 环境准备

建议 Python 3.10+。

```bash
conda create -n eval_llm_safety python=3.10 -y
conda activate eval_llm_safety

pip install -r requirements.txt
```

## 快速开始

### 1) 配置 `config.yaml`

至少确认以下字段：
- `model.name`：实验名（影响输出文件名前缀）
- `model.path`：被评测模型路径
- `model.mode`：当前代码主流程为 `vllm`
- `run_datasets`：本次要跑的数据集列表
- 对应 `datasets.<name>` 的 `dataset_path / prompt_column / eval` 配置

### 2) 直接运行

```bash
python3 main.py --config config.yaml
```

### 3) 用命令行覆盖配置（推荐）

```bash
python3 main.py \
  --config config.yaml \
  model.name="my-model" \
  model.path="/path/to/model" \
  run_datasets="['strongreject','wildjailbreak']" \
  api_config.eval_api_model="gpt-4o-mini" \
  api_config.max_workers=32 \
  datasets.strongreject.eval.evaluator_model="/path/to/Llama-Guard-3-8B"
```

### 4) 使用脚本 `run.sh`

`run.sh` 已给出完整示例（包含环境变量、日志目录和参数覆盖），可按实验需要修改：
- `MODEL_NAME`
- `MODEL_PATH`
- `RUN_DATASETS`
- `CUDA_VISIBLE_DEVICES`

## 数据集与评测方式

- `strongreject`
  - 推理输出：JSON
  - 评测方式：
    - `eval.eval_method: guard` -> `run_eval_guard_vllm`
    - `eval.eval_method: rubric` -> `run_eval_strongreject`
- `bsa`
  - 推理输出：CSV
  - 评测方式：`run_eval_bsa`，并调用 `utils/evaluate.py` 计算指标
- `xstest`
  - 推理输出：CSV
  - 评测方式：`run_eval_xstest`，附带分析文本与图表
- `wildchat` / `wildjailbreak`
  - 推理输出：JSON
  - 评测方式：`run_eval_guard_vllm`
- `oktest` / `phtest` / `falsereject` / `xstest-or`
  - 推理输出：JSON
  - 评测方式：`run_eval_frr`

## 输出说明

一次完整运行后，典型产物包括：
- `outputs/infer/<dataset>/<model>_<dataset>_(w|wo)_template.(json|csv)`
- `outputs/eval/<dataset>/...`（各评测器生成的中间文件）
- `outputs/result/<model>_summary.txt`（跨数据集摘要）

其中 `summary` 会按数据集追加写入，例如：
- `--- STRONGREJECT_W_TEMPLATE Summary ---`
- `--- XSTEST_WO_TEMPLATE Summary ---`

## 常见配置项

- `datasets.<name>.use_template`
  - 是否启用模板包装（由 `apply_template` 控制）
- `datasets.<name>.use_tokenizer_template`
  - 是否使用 tokenizer 的 `apply_chat_template`
- `datasets.<name>.sampling_params`
  - `temperature / top_p / max_tokens / n`
- `datasets.<name>.dataset_specific_params.pass_k`
  - 用于覆盖某些数据集生成数（例如 BSA）

## 日志与复现实验

- `run.sh` 会创建：
  - `EXPERIMENT_LOG_DIR=experiment_logs/eval_<model>_<time>/`
  - 运行日志：`eval.log`
- API 相关日志默认在 `api_logs/`

## 注意事项

- 当前仓库中的配置/代码里存在示例 API Key，请在私有环境中改为你自己的密钥，并避免提交到公共仓库。
- 若评测流程使用 API Judge，请确保网络与配额可用。
- 若使用 Llama-Guard 评测，请确认 `eval.evaluator_model` 路径正确，且 GPU 显存足够。

## 扩展新数据集（最小改动路径）

1. 在 `config.yaml` 增加 `datasets.<new_dataset>` 配置。
2. 在 `utils/data_handler.py` 的 `load_and_prep_data` 增加加载逻辑。
3. 在 `utils/eval_engine.py` 增加 `run_eval_<new_dataset>` 或复用现有评测函数。
4. 在 `main.py` 的阶段 3 分发逻辑中注册新数据集。