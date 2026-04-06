# 🛡️ LLM Safety Alignment Evaluation Pipeline

本项目是一个高效、可扩展且基于配置驱动的大语言模型（LLM）安全对齐评测框架。项目旨在通过统一的流水线完成大规模数据的自动化推理与 LLM-as-a-Judge 评估，彻底解决显存溢出 (OOM) 和多模型调度效率低下的问题。

## ✨ 核心特性

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

## ⚙️ 环境依赖
确保你的环境中安装了以下基础依赖：
```text
conda create -n eval_llm_safety python=3.10
conda activate eval_llm_safety

pip install -r requirements.txt
```

## 🚀 快速上手
方式 1：完全基于 YAML 运行
```bash
python main.py --config config.yaml
```

方式 2：使用启动脚本配合动态覆盖（推荐）
我们推荐使用 run.sh，这使得在不同模型或实验之间切换变得极其简单。

编辑 run.sh：
```bash
#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export HF_TOKEN="hf_你的token"

# 实验变量
MODEL_NAME="STAR1-R1-Distill-7B"
MODEL_PATH="/data/home/Yichen/data1/models/STAR1/STAR1-R1-Distill-7B"
EVALUATOR_MODEL="/data/home/Yichen/data1/models/meta-llama/Llama-Guard-3-8B"

# 动态覆盖 YAML 配置并运行
python3 main.py \
    --config config.yaml \
    model.name=$MODEL_NAME \
    model.path=$MODEL_PATH \
    run_datasets="['bsa', 'strongreject']" \
    api_config.eval_api_model="gpt-4o-mini" \
    api_config.max_workers=32 \
    datasets.strongreject.eval.evaluator_model=$EVALUATOR_MODEL
```

## 📈 输出示例
在所有评测运行结束后，你可以前往 outputs/result/{model_name}_summary.txt 查看高度格式化的对齐报告：

```Plaintext
--- STRONGREJECT Summary ---
                      ASR (Attack Success Rate) DSR (Defense Success Rate)
jailbreak                                                                 
none                                     11.82%                     88.18%
pair                                     14.06%                     85.94%
pap_misrepresentation                    10.22%                     89.78%

--- BSA Summary ---
          answer_at_1  answer_at_k  Cognitive Shortcut_think_at_1  ...  F-Score
model                                                                                                                                                                                                                                          
STAR1-7B        88.12         66.5                          26.88  ...     36.7

--- XSTEST Summary ---
Performance by Safe vs Unsafe Prompts
(Labels: 1=Compliance, 2=Refusal, 3=Partial Refusal)
final_label        1     2    3
type                           
Safe Prompts   59.6% 30.8% 9.6%
Unsafe Prompts 12.0% 80.5% 7.5%
```

---
## 🛠️ 如何添加一个新的评测集？

本框架采用极度解耦的设计，添加新数据集只需 **4 步**：

1. **配置**：在 `config.yaml` 的 `datasets` 节点下增加新数据集的默认参数。
2. **加载**：在 `utils/data_handler.py` 的 `load_and_prep_data` 中增加对应的数据集下载与解析逻辑。
3. **评测引擎**：在 `utils/eval_engine.py` 中编写专属的 `run_eval_xxx` 函数（如果只是算 ASR，可以直接复用已有的 `run_eval_guard_vllm`）。
4. **注册**：在 `main.py` 的 `阶段 3: 集中评测` 的 `if-elif` 语句中加入新数据集的方法分发。