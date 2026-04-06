import os
import sys
import json
import time
import pandas as pd
from typing import Optional, List, Dict, Any

from datasets import load_dataset, concatenate_datasets, Dataset

# 尝试导入 StrongReject 越狱库
try:
    from strong_reject.jailbreaks import apply_jailbreaks_to_dataset
except ImportError:
    apply_jailbreaks_to_dataset = None

DATA_DIR = "datasets"


def file_exists(file_path):
    return os.path.exists(file_path) and os.path.isfile(file_path)


def load_and_prep_data(ds_name: str, config: Dict[str, Any]) -> tuple[List[str], List[Any]]:
    """
    根据数据集名称和配置加载及预处理数据。
    """
    dataset_path = config.get('dataset_path', '')
    prompt_column = config.get('prompt_column', 'prompt')
    limit_num = config.get('limit_num', float('inf'))

    if limit_num == -1:
        limit_num = float('inf')

    # --- 1. StrongReject ---
    if ds_name == "strongreject":
        if apply_jailbreaks_to_dataset is None:
            print("Error: 'strong_reject' library not found. Please install it.", file=sys.stderr)
            sys.exit(1)

        if not os.path.exists(dataset_path):
            print(f"Warning: Cache file {dataset_path} not found. Attempting to download...", file=sys.stderr)
            data_url = "https://raw.githubusercontent.com/alexandrasouly/strongreject/main/strongreject_dataset/strongreject_dataset.csv"
            dataset_hf = load_dataset("csv", data_files=data_url)["train"]
            os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
            dataset_hf.to_csv(dataset_path, index=False)

        # 处理 API Key (越狱可能需要)
        api_conf = config.get('api_config', {})
        api_key = os.environ.get("OPENAI_API_KEY", api_conf.get('api_key', ''))
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
            if 'base_url' in api_conf:
                os.environ["OPENAI_BASE_URL"] = api_conf['base_url']

        dataset = load_dataset("csv", data_files=dataset_path, encoding='utf-8')["train"]
        limit_num_val = min(len(dataset), limit_num)
        if limit_num_val != float('inf'):
            dataset = dataset.select(range(int(limit_num_val)))

        jailbreak_types_str = config.get('dataset_specific_params', {}).get('jailbreak_type', 'none')
        jailbreak_types = [j.strip() for j in jailbreak_types_str.split(',') if j.strip()]

        print(f"Applying jailbreaks: {jailbreak_types}", file=sys.stderr)
        jailbroken_dataset = apply_jailbreaks_to_dataset(dataset, jailbreak_types, num_proc=4)

        prompts_list = list(jailbroken_dataset[prompt_column])
        metadata_list = list(jailbroken_dataset)
        return prompts_list, metadata_list

    # --- 2. BSA ---
    elif ds_name == "bsa":
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"BSA dataset file not found at: {dataset_path}")
        with open(dataset_path, 'r') as f:
            dataset = json.load(f)

        limit_num_val = min(len(dataset), limit_num)
        if limit_num_val != float('inf'):
            dataset = dataset[:int(limit_num_val)]

        prompts_list = [item[prompt_column] for item in dataset]
        metadata_list = [item for item in dataset]
        return prompts_list, metadata_list

    # --- 3. XSTest ---
    elif ds_name == "xstest":
        if not os.path.exists(dataset_path):
            print(f"Warning: Cache file {dataset_path} not found. Downloading...", file=sys.stderr)
            data = load_dataset("csv",
                                data_files="https://raw.githubusercontent.com/paul-rottger/xstest/refs/heads/main/xstest_prompts.csv")[
                "train"]
            data = data.filter(lambda x: x["label"] == "safe")
            data = data.map(lambda x: {"question": x.pop("prompt")})
            df = data.to_pandas()
            os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
            df.to_csv(dataset_path, index=False)

        df = pd.read_csv(dataset_path)
        limit_num_val = min(len(df), limit_num)
        if limit_num_val != float('inf'):
            df = df.head(int(limit_num_val))

        prompts_list = df[prompt_column].tolist()
        metadata_list = df.to_dict('records')
        return prompts_list, metadata_list

    # --- 4. WildChat ---
    elif ds_name == "wildchat":
        local_data_path = "/data/home/Yichen/CC-GRPO/eval_llm_safety/datasets/wildchat_data"
        if file_exists(dataset_path):
            data = Dataset.from_pandas(pd.read_csv(dataset_path))
        else:
            print(f"Downloading and processing WildChat...", file=sys.stderr)
            data = load_dataset(local_data_path)["train"]
            data = data.filter(lambda x: x["toxic"] == True and x["turn"] == 1 and x["language"] == "English" and
                                         x["openai_moderation"][-1]["flagged"] == True)
            data = data.map(
                lambda x: {"category": (k := max(d := x.pop("openai_moderation")[-1]["category_scores"], key=d.get)),
                           "category_score": d[k], "question": x.pop("conversation")[0]["content"]}
            )
            ds_all = []
            harmful_types = ["harassment", "harassment/threatening", "hate", "hate/threatening", "self-harm",
                             "self-harm/instructions", "self-harm/intent", "sexual", "sexual/minors", "violence",
                             "violence/graphic"]
            for harmful_type in harmful_types:
                ds_filter = data.filter(lambda x: x["category"] == harmful_type)
                sorted_ds_filter = ds_filter.sort("category_score", reverse=True)
                top_100_ds_filter = sorted_ds_filter.select(range(min(100, len(sorted_ds_filter))))
                ds_all.append(top_100_ds_filter)

            data = concatenate_datasets(ds_all)
            os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
            data.to_pandas().to_csv(dataset_path, index=False)

        limit_num_val = min(len(data), limit_num) if limit_num != float('inf') else len(data)
        if limit_num != float('inf'):
            data = data.select(range(int(limit_num_val)))

        prompts_list = data[prompt_column]
        metadata_list = list(data)
        return prompts_list, metadata_list

    # --- 5. WildJailbreak ---
    elif ds_name == "wildjailbreak":
        if file_exists(dataset_path):
            data = Dataset.from_pandas(pd.read_csv(dataset_path))
        else:
            print(f"Downloading and processing WildJailbreak...", file=sys.stderr)
            data = load_dataset("allenai/wildjailbreak", name="eval",
                                cache_dir=os.path.join(DATA_DIR, "wildjailbreak_temp_hf_cache"))["train"]
            data = data.map(lambda x: {"source": "wildjailbreak", "question": x.pop("adversarial")})
            data = data.filter(lambda x: x["data_type"] == "adversarial_harmful")
            data = data.shuffle(seed=42).select(range(500))
            os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
            data.to_pandas().to_csv(dataset_path, index=False)

        limit_num_val = min(len(data), limit_num) if limit_num != float('inf') else len(data)
        if limit_num != float('inf'):
            data = data.select(range(int(limit_num_val)))

        prompts_list = data[prompt_column]
        metadata_list = list(data)
        return prompts_list, metadata_list
    else:
        raise ValueError(f"未知的数据集类型: {ds_name}")


def process_and_save_results(ds_name: str, model_name: str, output_dir: str, config: Dict[str, Any],
                             original_prompts_list: List[str], final_prompts: List[str],
                             raw_outputs_nested: List[List[str]], metadata_list: List[Any]) -> str:
    """
    统一保存推理结果的逻辑。
    """
    output_file = ""
    min_len = min(len(raw_outputs_nested), len(metadata_list), len(original_prompts_list))
    all_results = []

    # --- 1. JSON 输出 (StrongReject, Wildchat, Wildjailbreak) ---
    if ds_name in ["strongreject", "wildchat", "wildjailbreak"]:
        for idx in range(min_len):
            original_data_dict = dict(metadata_list[idx])
            original_data_dict["instruction"] = original_prompts_list[idx]
            original_data_dict["prompt"] = final_prompts[idx]
            original_data_dict["response"] = raw_outputs_nested[idx]
            original_data_dict["model"] = model_name
            all_results.append(original_data_dict)

        output_file = os.path.join(output_dir, f"{model_name}.json")
        meta_data = {"model_name": model_name, "dataset": ds_name}
        saved_data = {'meta_info': meta_data, 'data': all_results}

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(saved_data, f, ensure_ascii=False, indent=4, default=str)

    # --- 2. CSV 输出 (BSA) ---
    elif ds_name == "bsa":
        for idx in range(min_len):
            for response_text in raw_outputs_nested[idx]:
                all_results.append({
                    'prompt': final_prompts[idx],
                    'response': response_text,
                    'id': metadata_list[idx].get('id', 'N/A')
                })
        output_file = os.path.join(output_dir, f"{model_name}.csv")
        pd.DataFrame(all_results).to_csv(output_file, index=False, encoding='utf-8')

    # --- 3. CSV 输出 (XSTest) ---
    elif ds_name == "xstest":
        for i in range(min_len):
            item = metadata_list[i]
            for response_text in raw_outputs_nested[i]:
                all_results.append({
                    'id': item.get('id', i),
                    'prompt': original_prompts_list[i],
                    'completion': response_text,
                    'type': item.get('type', 'N/A'),
                    'label': item.get('label', 'N/A'),
                    'model': model_name
                })
        output_file = os.path.join(output_dir, f"{model_name}.csv")
        pd.DataFrame(all_results).to_csv(output_file, index=False, encoding='utf-8')

    print(f"Results saved to {output_file}", file=sys.stderr)
    return output_file