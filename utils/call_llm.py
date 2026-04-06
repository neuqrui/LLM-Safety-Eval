import re
import time
import random
import multiprocessing as mp
from multiprocessing import Process, Queue
from concurrent.futures import ProcessPoolExecutor, TimeoutError, ThreadPoolExecutor
import threading
import logging
import anthropic
from openai import OpenAI
from typing import List, Dict, Optional
from datetime import datetime
import os
import torch
import gc
import sys

from tqdm import tqdm

api_url = "https://api.chatanywhere.tech/v1"
key1 = "sk-v221pe0zDxwtK8JVXlXNCSXDwUVPhM4YOo8pcxvIvuuEZzTk"
key2 = "sk-bRDBciepkuKwh6j1fZ8qc7kEOzB5fKHFZXT0ysAk6zXLEkx4"
key3 = "sk-qM4CrDOHDqCtPY6lknfgGKc90U4xvbjHHGStXmVy7Gs18PeS"
key4 = "sk-D9cEfgAjOatvMW3leb9ESvujvaiGY0NxMYrM7FhdZvEC9sGC"


def log_message(msg, timestamp, log_dir='logs', file_prefix='log_rm_example'):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'{file_prefix}.txt')

    message = f"[{timestamp}] {msg}"

    with open(log_file, 'a', encoding='utf-8') as f:
        f"[{timestamp}] {message}"
        f.write(message + '\n')

    return log_file  # 可选：返回写入的文件名


class APIPool:

    def __init__(self, api_keys: List[str], max_error_count: int = 10, log_dir: str = None,
                 retry_interval_minutes: int = 30):
        """
        Args:
            api_keys: List of API keys
            max_error_count: Maximum number of errors allowed for an API key, beyond which it will be marked as unavailable
            log_dir: Log 目录
            retry_interval_minutes: API restart check interval (minutes), default is 30 minutes
        """
        """
        Examples:
            self.api_pool = APIPool(api_keys, max_error_count=10)
            self.base_url = api_url 
        """
        self.api_keys = api_keys
        self.max_error_count = max_error_count
        self.error_counts: Dict[str, int] = {key: 0 for key in api_keys}
        self.available_keys = set(api_keys)
        self.unavailable_timestamps: Dict[str, float] = {}  # Record timestamps when API keys are marked as unavailable
        self.retry_interval_seconds = retry_interval_minutes * 60  # Convert to seconds

        if log_dir is None:
            self.log_dir = os.path.join(os.getcwd(), 'api_logs')
        else:
            self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

        # Log file path
        self.log_file = os.path.join(self.log_dir, f'api_pool_{datetime.now().strftime("%Y%m%d")}.log')

        # Initialize log
        self._log(f"API pool initialized with {len(api_keys)} API keys")
        for key in api_keys:
            self._log(f"Added API key: {self._mask_api_key(key)}")

    def _mask_api_key(self, api_key: str) -> str:
        # 只输出前四位 and 后四位
        if len(api_key) <= 8:
            return api_key
        return f"{api_key[:4]}...{api_key[-4:]}"

    def _log(self, message: str, is_error: bool = False):
        # 将错误记录到log文件并未在控制台输出，只记录错误
        if is_error:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_message = f"[{timestamp}] {message}"

            # Write to log file
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(log_message + '\n')

    def get_api_key(self) -> Optional[str]:
        # 取出一个api key 并且检查被占用的api key是否可以重新启动
        """
        Get an available API key and check if unavailable API keys can be re-enabled

        Returns:
            Available API key, or None if no API key is available
        """
        # Check if any unavailable API keys can be re-enabled
        current_time = time.time()
        keys_to_check = []

        # 满足retry 时间间隔的key，需要被检查
        for api_key, timestamp in list(self.unavailable_timestamps.items()):
            if current_time - timestamp >= self.retry_interval_seconds:
                keys_to_check.append(api_key)

        # Recheck these API keys
        for api_key in keys_to_check:
            self._log(
                f"Rechecking API key: {self._mask_api_key(api_key)}, {self.retry_interval_seconds // 60} minutes have passed")
            # Reset error count and add API key back to available list
            self.reset_error_count(api_key)
            # Remove from unavailable timestamps dictionary
            self.unavailable_timestamps.pop(api_key, None)

        if not self.available_keys:
            self._log("Warning: No available API keys", is_error=True)
            return None

        # Randomly select an available API key
        api_key = random.choice(list(self.available_keys))
        return api_key

    def mark_error(self, api_key: str, error_message: str):
        """
        Mark an API key as having an error

        Args:
            api_key: API key with error
            error_message: Error information
        """
        if api_key not in self.api_keys:
            self._log(f"Warning: Attempting to mark unknown API key: {self._mask_api_key(api_key)}", is_error=True)
            return

        # Increase error count
        self.error_counts[api_key] += 1
        current_count = self.error_counts[api_key]

        self._log(
            f"API key {self._mask_api_key(api_key)} encountered an error ({current_count}/{self.max_error_count}): {error_message}",
            is_error=True)

        # If error count exceeds threshold, mark as unavailable
        if current_count >= self.max_error_count and api_key in self.available_keys:
            self.available_keys.remove(api_key)
            # Record timestamp when API key is marked as unavailable
            self.unavailable_timestamps[api_key] = time.time()
            self._log(
                f"API key {self._mask_api_key(api_key)} has been marked as unavailable, reached maximum error count {self.max_error_count}, will be rechecked in {self.retry_interval_seconds // 60} minutes",
                is_error=True)

    def reset_error_count(self, api_key: str):
        if api_key not in self.api_keys:
            self._log(f"Warning: Attempting to reset unknown API key: {self._mask_api_key(api_key)}", is_error=True)
            return

        # Reset error count
        self.error_counts[api_key] = 0

        if api_key not in self.available_keys:
            self.available_keys.add(api_key)

    def get_status(self) -> Dict:
        current_time = time.time()
        unavailable_info = {}

        for api_key, timestamp in self.unavailable_timestamps.items():
            elapsed_seconds = current_time - timestamp  # 已经停用的时间
            remaining_seconds = max(0, self.retry_interval_seconds - elapsed_seconds)  # 剩余停用时间
            unavailable_info[self._mask_api_key(api_key)] = {
                "elapsed_minutes": round(elapsed_seconds / 60, 1),
                "remaining_minutes": round(remaining_seconds / 60, 1)
            }

        return {
            "total_keys": len(self.api_keys),
            "available_keys": len(self.available_keys),
            "unavailable_keys": len(self.api_keys) - len(self.available_keys),
            "error_counts": {self._mask_api_key(k): v for k, v in self.error_counts.items()},
            "unavailable_info": unavailable_info
        }

api_pool = APIPool(api_keys=[key1, key2, key3, key4], max_error_count=5)

def call_api_for_inference(constructed_input: str, api_type: str, api_name=None, system_prompt: Optional[str] = None) -> str:
    """
    Call Ali Baidu API for inference

    Args:
        question: prompt content(question + category + response)

    Returns:
        Answer from the API
    """
    assert constructed_input is not None
    max_retries = min(3, len(api_pool.available_keys))
    retry_count = 0
    if api_name is None:
        if api_type == 'openai':
            api_name = "gpt-3.5-turbo"
        elif api_type == 'anthropic':
            api_name = "claude-3-5-sonnet-20241022"  # 'gemini-2.5-flash-preview'

    while retry_count < max_retries:
        api_key = api_pool.get_api_key()
        if api_key is None:
            print("No available API key, cannot call API")
            return ""
        try:
            if api_type == 'openai':
                messages = []
                if system_prompt:  # 如果 system_prompt 存在且非空
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": constructed_input})

                with OpenAI(api_key=api_key, base_url=api_url) as client:
                    response = client.chat.completions.create(
                        model=api_name,
                        messages=messages
                    )
                    reward_content = response.choices[0].message.content
            elif api_type == 'anthropic':
                with anthropic.Anthropic(api_key=api_key, base_url=api_url) as client:
                    response = client.messages.create(
                        model=api_name,
                        max_tokens=500,
                        temperature=0.6,
                        messages=[{"role": "user", "content": constructed_input}]
                    )
                    reward_content = response.content[0].text
            else:
                assert 0

            return reward_content

        except Exception as e:
            error_message = str(e)
            api_pool.mark_error(api_key, error_message)

            # Check error type, determine if retry is needed
            if "余额不足" in error_message or "Balance is insufficient" in error_message or "not sufficient to support" in error_message:
                print(f"API key balance insufficient: {error_message}")
                # Continue to try the next API key
                retry_count += 1
                continue
            elif "rate_limit_exceeded" in error_message:
                print(f"API call frequency limit exceeded: {error_message}")
                # Wait for a while and retry
                time.sleep(2)
                retry_count += 1
                continue
            else:
                print(f"API call failed: {error_message}")
                # Other errors, also try to retry
                retry_count += 1
                continue

    print(f"Tried {max_retries} times API call, all failed")
    return ""


def call_llm_parallel(prompt: List, max_workers: int, api_type="openai", api_name="gpt-4o-mini", system_prompt: Optional[str] = None):
    future_to_idx = {}
    api_answers = [None] * len(prompt)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:

        for idx, text in enumerate(prompt):
            future_to_idx[executor.submit(call_api_for_inference, text, api_type, api_name, system_prompt)] = (idx, text)

        for future in tqdm(future_to_idx, desc="Parallel call api"):
            idx, text = future_to_idx[future]
            try:
                res_content = future.result()
                api_answers[idx] = res_content
            except Exception as e:
                print(f"Error occurred while processing item {idx}: {e}")

    return api_answers


def call_api_for_xstest(prompt_text: str, api_name: str) -> str:
    """
    专门为 XSTest 调用 OpenAI API，使用其特定的参数。
    """
    max_retries = min(3, len(api_pool.available_keys))
    retry_count = 0

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt_text}
    ]

    while retry_count < max_retries:
        api_key = api_pool.get_api_key()
        if api_key is None:
            assert 0
        try:
            with OpenAI(api_key=api_key, base_url=api_url) as client:
                response = client.chat.completions.create(
                    model=api_name,
                    messages=messages,
                    # --- XSTest 特定的采样参数 ---
                    temperature=0.0,
                    max_tokens=16,
                    top_p=1.0,
                    frequency_penalty=0.0,
                    presence_penalty=0.0
                )
                reward_content = response.choices[0].message.content.strip()
                api_pool.reset_error_count(api_key)
                return reward_content
        except Exception as e:
            error_message = str(e)
            api_pool.mark_error(api_key, error_message)
            print(f"XSTest API call failed (try {retry_count + 1}/{max_retries}): {error_message}", file=sys.stderr)

            if "rate_limit_exceeded" in error_message:
                wait_time = 2 * (retry_count + 1)  # 指数退避
                print(f"Rate limit exceeded, waiting {wait_time}s...", file=sys.stderr)
                time.sleep(wait_time)
            elif "余额不足" in error_message or "Balance is insufficient" in error_message or "not sufficient to support" in error_message:
                print(f"API key balance insufficient, trying next key...", file=sys.stderr)
                # 不需要等待，直接进入下一次循环尝试新 key
            else:
                time.sleep(1)  # 其他错误，稍微等待

            retry_count += 1

    print(f"Tried {max_retries} times for XSTest API call, all failed for prompt: {prompt_text[:50]}...",
          file=sys.stderr)
    return "ERROR_API_FAILED_RETRIES"  # 返回错误标识


# --- 新增：专门为 XSTest 定制的并行调用函数 ---
def call_llm_parallel_xstest(prompts: List[str], max_workers: int, api_name: str) -> List[str]:
    """
    使用线程池并行调用 call_api_for_xstest 函数。
    """
    future_to_idx = {}
    api_answers = [None] * len(prompts)

    # 确保 api_pool 存在 (假设它在文件顶部已初始化)
    if 'api_pool' not in globals():
        print("Error: APIPool not initialized.", file=sys.stderr)
        return ["ERROR_API_POOL_MISSING"] * len(prompts)

    print(f"Starting parallel XSTest API calls with {max_workers} workers...", file=sys.stderr)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for idx, text in enumerate(prompts):
            future_to_idx[executor.submit(call_api_for_xstest, text, api_name)] = (idx, text[:50])  # 记录部分文本用于调试

        for future in tqdm(future_to_idx, desc="Parallel XSTest API calls"):
            idx, text_preview = future_to_idx[future]
            try:
                res_content = future.result()
                api_answers[idx] = res_content
            except Exception as e:
                print(f"Error occurred while processing XSTest item {idx} ('{text_preview}...'): {e}", file=sys.stderr)
                api_answers[idx] = f"ERROR_FUTURE_FAILED: {e}"

    print(f"Finished parallel XSTest API calls.", file=sys.stderr)
    return api_answers