#!/usr/bin/env python3
"""Fill empty <available_controls>[] entries in wm_train.json with CloudGPT/GOT.

For each sample whose assistant response has an empty available_controls list,
the script infers the current pair directory, sends the current screenshot
(pair_N/prev.png), current controls from the prompt, and next screenshot
(pair_N/next.png) to a vision-capable model, then replaces only the controls
JSON in the assistant response.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from tqdm import tqdm


WM_EVAL_CODE = "/local_nvme/zhanglechao/wm_eval/eval_code"
if WM_EVAL_CODE not in sys.path:
    sys.path.insert(0, WM_EVAL_CODE)

from cloudgpt_aoai import encode_image, get_openai_client  # noqa: E402


EMPTY_CONTROLS_RE = re.compile(
    r"<available_controls>\s*\[\s*\]\s*</available_controls>", re.DOTALL
)
CONTROLS_BLOCK_RE = re.compile(
    r"(<available_controls>\s*)(.*?)(\s*</available_controls>)", re.DOTALL
)
CURRENT_CONTROLS_RE = re.compile(
    r"Current Available Controls:\s*\n(?P<controls>\[[\s\S]*?\])\s*\n\s*Action:",
    re.DOTALL,
)
ACTION_RE = re.compile(
    r"Action:\s*\n(?P<action>\{[\s\S]*?\})\s*\n\s*GUI Action Description:",
    re.DOTALL,
)
DESC_RE = re.compile(
    r"<next_state_description>\s*(?P<desc>[\s\S]*?)\s*</next_state_description>",
    re.DOTALL,
)
STEP_RE = re.compile(r"--- Step \d+ ---")
PAIR_RE = re.compile(r"^(pair_)(\d+)$")


SYSTEM_PROMPT = (
    "You are a precise Office UI accessibility annotator. "
    "Return only the JSON array requested by the user."
)


def parse_json_array(text: str) -> list[dict[str, Any]]:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    match = re.search(r"\[[\s\S]*\]", text)
    if not match:
        raise ValueError("model output does not contain a JSON array")

    data = json.loads(match.group(0))
    if not isinstance(data, list):
        raise ValueError("model output JSON is not a list")

    cleaned = []
    for item in data:
        if not isinstance(item, dict):
            continue
        cleaned.append(
            {
                "control_type": str(item.get("control_type", "")),
                "control_text": str(item.get("control_text", "")),
            }
        )
    if not cleaned:
        raise ValueError("model returned an empty controls list")
    return cleaned[:100]


def extract_current_controls(user_content: str) -> list[dict[str, Any]]:
    match = CURRENT_CONTROLS_RE.search(user_content)
    if not match:
        return []
    try:
        data = json.loads(match.group("controls"))
    except json.JSONDecodeError:
        return []
    if isinstance(data, list):
        return data
    return []


def extract_action(user_content: str) -> dict[str, Any] | None:
    match = ACTION_RE.search(user_content)
    if not match:
        return None
    try:
        return json.loads(match.group("action"))
    except json.JSONDecodeError:
        return None


def extract_description(assistant_content: str) -> str:
    match = DESC_RE.search(assistant_content)
    return match.group("desc").strip() if match else ""


def history_count(user_content: str) -> int:
    return len(STEP_RE.findall(user_content))


def infer_pair_paths(sample: dict[str, Any]) -> tuple[str, str]:
    images = sample.get("images") or []
    if not images:
        raise ValueError("sample has no images")

    anchor = Path(images[0])
    pair_dir = anchor.parent
    task_dir = pair_dir.parent
    pair_match = PAIR_RE.match(pair_dir.name)
    if not pair_match:
        raise ValueError(f"cannot infer pair dir from {anchor}")

    prefix, num = pair_match.groups()
    width = len(num)
    user_content = sample["messages"][1]["content"]
    current_num = history_count(user_content) + 1
    current_pair = task_dir / f"{prefix}{current_num:0{width}d}"
    prev_img = current_pair / "prev.png"
    next_img = current_pair / "next.png"
    if not prev_img.exists():
        raise FileNotFoundError(prev_img)
    if not next_img.exists():
        raise FileNotFoundError(next_img)
    return str(prev_img), str(next_img)


def build_prompt(sample: dict[str, Any], index: int) -> str:
    user_content = sample["messages"][1]["content"]
    assistant_content = sample["messages"][-1]["content"]
    controls = extract_current_controls(user_content)
    action = extract_action(user_content)
    description = extract_description(assistant_content)

    return f"""Fill the missing next-state available UI controls for one Office UI transition.

You are given two screenshots:
1. Image 1: the current/previous UI state before the action.
2. Image 2: the next UI state after the action.

Use Image 2 as the main evidence for controls in the next state. Use the current controls as a prior: keep controls that remain visible/available, remove controls that disappear, and add visible new controls from the next screenshot. Return only a JSON array. Each item must have exactly these keys: "control_type" and "control_text".

Dataset index: {index}

Action:
{json.dumps(action, ensure_ascii=False, indent=2) if action is not None else "null"}

Current Available Controls:
{json.dumps(controls, ensure_ascii=False, indent=2)}

Known next-state visual description:
{description}

Output only JSON, with no markdown and no explanation.
"""


def call_model(client: Any, model: str, sample: dict[str, Any], index: int, max_retries: int) -> list[dict[str, Any]]:
    prev_img, next_img = infer_pair_paths(sample)
    prompt = build_prompt(sample, index)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Image 1: current/previous UI state."},
                {"type": "image_url", "image_url": {"url": encode_image(prev_img)}},
                {"type": "text", "text": "Image 2: next UI state whose controls must be listed."},
                {"type": "image_url", "image_url": {"url": encode_image(next_img)}},
                {"type": "text", "text": prompt},
            ],
        },
    ]

    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                max_completion_tokens=4096,
            )
            content = (resp.choices[0].message.content or "").strip()
            return parse_json_array(content)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt < max_retries:
                time.sleep(5 * attempt)
    raise RuntimeError(f"model failed after {max_retries} attempts: {last_error}")


def replace_controls(assistant_content: str, controls: list[dict[str, Any]]) -> str:
    controls_text = json.dumps(controls, ensure_ascii=False, indent=2)
    updated, count = CONTROLS_BLOCK_RE.subn(
        lambda match: f"{match.group(1)}{controls_text}{match.group(3)}",
        assistant_content,
        count=1,
    )
    if count != 1:
        raise ValueError("assistant content has no available_controls block")
    return updated


def load_progress(path: str | None) -> dict[str, list[dict[str, Any]]]:
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {str(k): v for k, v in raw.items()}


def save_progress(path: str | None, progress: dict[str, list[dict[str, Any]]]) -> None:
    if not path:
        return
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="/local_nvme/zhanglechao/LLaMA-Factory/data/wm_train.json")
    parser.add_argument("--output", default="/local_nvme/zhanglechao/LLaMA-Factory/data/wm_train_got55_fill.json")
    parser.add_argument("--progress", default="/local_nvme/zhanglechao/LLaMA-Factory/data/wm_train_got55_fill_progress.json")
    parser.add_argument("--model", default="gpt-5.2-chat-20251211")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    empty_indices = [
        i for i, sample in enumerate(data)
        if EMPTY_CONTROLS_RE.search(sample["messages"][-1]["content"])
    ]
    if args.limit is not None:
        empty_indices = empty_indices[: args.limit]

    print(f"input={args.input}")
    print(f"output={args.output}")
    print(f"model={args.model}")
    print(f"empty samples selected={len(empty_indices)}")

    if args.dry_run:
        for idx in empty_indices[:5]:
            prev_img, next_img = infer_pair_paths(data[idx])
            controls = extract_current_controls(data[idx]["messages"][1]["content"])
            print(f"idx={idx} prev={prev_img} next={next_img} current_controls={len(controls)}")
        return

    progress = load_progress(args.progress)
    failures: dict[str, str] = {}
    done_since_save = 0
    pending_indices = [idx for idx in empty_indices if str(idx) not in progress]

    def fill_one(idx: int) -> tuple[int, list[dict[str, Any]]]:
        client = get_openai_client()
        return idx, call_model(client, args.model, data[idx], idx, args.max_retries)

    if args.workers <= 1:
        for idx in tqdm(pending_indices, desc="Filling empty controls"):
            key = str(idx)
            try:
                _, controls = fill_one(idx)
                progress[key] = controls
                done_since_save += 1
                if done_since_save >= args.save_every:
                    save_progress(args.progress, progress)
                    done_since_save = 0
            except Exception as exc:  # noqa: BLE001
                failures[key] = str(exc)
                print(f"[FAIL] idx={idx}: {exc}")
                save_progress(args.progress, progress)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(fill_one, idx): idx for idx in pending_indices}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Filling empty controls"):
                idx = futures[future]
                key = str(idx)
                try:
                    _, controls = future.result()
                    progress[key] = controls
                    done_since_save += 1
                    if done_since_save >= args.save_every:
                        save_progress(args.progress, progress)
                        done_since_save = 0
                except Exception as exc:  # noqa: BLE001
                    failures[key] = str(exc)
                    print(f"[FAIL] idx={idx}: {exc}")
                    save_progress(args.progress, progress)

    save_progress(args.progress, progress)

    filled = copy.deepcopy(data)
    for key, controls in progress.items():
        idx = int(key)
        filled[idx]["messages"][-1]["content"] = replace_controls(
            filled[idx]["messages"][-1]["content"], controls
        )

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(filled, f, ensure_ascii=False, indent=2)

    remaining_empty = sum(
        1 for sample in filled
        if EMPTY_CONTROLS_RE.search(sample["messages"][-1]["content"])
    )
    print(f"filled={len(progress)} failures={len(failures)} remaining_empty={remaining_empty}")
    if failures:
        fail_path = f"{args.output}.failures.json"
        with open(fail_path, "w", encoding="utf-8") as f:
            json.dump(failures, f, ensure_ascii=False, indent=2)
        print(f"failures saved to {fail_path}")


if __name__ == "__main__":
    main()
