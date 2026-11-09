import ast
import hashlib
import json
import keyword
import random
import re
from collections import Counter
from pathlib import Path
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parent
MODEL = "Qwen/Qwen3.5-2B"
MODEL_REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"
DATA_REVISION = "6bda777c88d21e5a204703c1ee45597a8fa4f734"
SEED = 42


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def schema_json(value):
    if isinstance(value, list):
        return [schema_json(x) for x in value]
    if not isinstance(value, dict):
        return value
    out = {k: schema_json(v) for k, v in value.items()}
    if isinstance(out.get("type"), str):
        out["type"] = {"dict": "object", "float": "number", "int": "integer",
                       "bool": "boolean", "str": "string", "list": "array",
                       "tuple": "array", "any": "string"}.get(out["type"], out["type"])
    return out


def literal(node):
    if isinstance(node, ast.Name) and node.id in {"true", "false", "null"}:
        return {"true": True, "false": False, "null": None}[node.id]
    if isinstance(node, (ast.List, ast.Tuple)):
        return [literal(x) for x in node.elts]
    if isinstance(node, ast.Dict):
        return {literal(k): literal(v) for k, v in zip(node.keys, node.values)}
    return ast.literal_eval(node)


def read_calls(text, names):
    text = text.strip()
    if text.startswith("<"):
        raise ValueError("unsupported_reference_markup")
    if not text.startswith("["):
        return []
    aliases = {name: f"tool_{i}" for i, name in enumerate(names)}
    pattern = re.compile("|".join(re.escape(name) for name in sorted(names, key=len, reverse=True)) + r"|\b(?:" + "|".join(keyword.kwlist) + r")\b")
    parts = re.split(r'''("(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*')''', text)
    for i in range(0, len(parts), 2):
        def replace(match):
            word = match.group()
            tail = parts[i][match.end():].lstrip()
            if word in aliases and tail.startswith("("):
                return aliases[word]
            if word in keyword.kwlist and tail.startswith("="):
                return "arg_" + word
            return word
        parts[i] = pattern.sub(replace, parts[i])
    expr = ast.parse("".join(parts), mode="eval").body
    if not isinstance(expr, ast.List):
        raise ValueError("expected a list of calls")
    reverse = {v: k for k, v in aliases.items()}
    calls = []
    for node in expr.elts:
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name) or node.args:
            raise ValueError("unsupported reference call")
        args = {}
        for arg in node.keywords:
            key = arg.arg
            if key is None:
                raise ValueError("kwargs expansion in reference")
            if key.startswith("arg_") and key[4:] in keyword.kwlist:
                key = key[4:]
            args[key] = literal(arg.value)
        calls.append({"name": reverse[node.func.id], "arguments": args})
    return calls


def unpack(row, dialog_id):
    marker = "Here is a list of functions in JSON format that you can invoke:"
    if marker not in row["system"]:
        raise ValueError("noncanonical_tool_schema")
    raw_tools = json.JSONDecoder().raw_decode(row["system"].split(marker, 1)[1].lstrip())[0]
    names = [tool["name"] for tool in raw_tools]
    renamed = {}
    used = set()
    tools = []
    for tool in raw_tools:
        name = re.sub(r"[^a-zA-Z0-9_.-]+", "_", tool["name"]).strip("_") or "tool"
        if name in used:
            name += "_" + str(len(used))
        used.add(name)
        renamed[tool["name"]] = name
        tools.append({"type": "function", "function": {
            "name": name, "description": tool.get("description", ""),
            "parameters": schema_json(tool["parameters"])
        }})

    time_match = re.search(r"The current time is ([\d :.-]+)", row["system"])
    messages = [{"role": "system", "content": "Current time: " + time_match[1]}] if time_match else []
    examples = []
    for turn, message in enumerate(row["conversations"]):
        role, text = message["from"], message["value"]
        if role == "assistant":
            calls = read_calls(text, names)
            calls = [{**call, "name": renamed[call["name"]]} for call in calls]
            if messages and messages[-1]["role"] == "user":
                examples.append({"id": f"{dialog_id}_{turn}", "dialog_id": dialog_id,
                                 "messages": list(messages), "tools": tools, "reference": calls,
                                 "reference_text": text, "kind": "call" if calls else "no_call"})
            if calls:
                messages.append({"role": "assistant", "content": "", "tool_calls": [
                    {"type": "function", "function": call} for call in calls
                ]})
            else:
                messages.append({"role": "assistant", "content": text})
        else:
            messages.append({"role": role, "content": text})
    return examples


def main():
    from transformers import AutoTokenizer
    import numpy as np

    data_dir = ROOT / "data"
    data_dir.mkdir(exist_ok=True)
    raw_path = data_dir / "toolace.json"
    if not raw_path.exists():
        url = f"https://huggingface.co/datasets/Team-ACE/ToolACE/resolve/{DATA_REVISION}/data.json"
        with urlopen(url) as response:
            raw_path.write_bytes(response.read())
    raw = json.loads(raw_path.read_text())
    benchmark = [json.loads(line) for line in (ROOT / "bfcl.jsonl").read_text().splitlines()]
    benchmark_queries = {" ".join(m["content"].casefold().split()) for ex in benchmark
                         for m in ex["question"] if m["role"] == "user"}
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REVISION)
    groups = list(range(len(raw)))
    random.Random(SEED).shuffle(groups)
    val_ids = set(groups[:565])
    rows = {"train": [], "validation": []}
    rejected = []
    seen = {}
    lengths = []
    tool_names = set()
    for dialog_id, dialog in enumerate(raw):
        split = "validation" if dialog_id in val_ids else "train"
        try:
            examples = unpack(dialog, dialog_id)
        except (ValueError, KeyError, SyntaxError, TypeError) as e:
            rejected.append({"dialog_id": dialog_id, "reason": type(e).__name__ + ": " + str(e)})
            continue
        for ex in examples:
            last_user = ex["messages"][-1]["content"]
            if not last_user.strip() or not ex["reference_text"].strip():
                rejected.append({"id": ex["id"], "reason": "empty_text"})
                continue
            if " ".join(last_user.casefold().split()) in benchmark_queries:
                rejected.append({"id": ex["id"], "reason": "bfcl_question_overlap"})
                continue
            prompt = tokenizer.apply_chat_template(ex["messages"], tools=ex["tools"], tokenize=False,
                                                   add_generation_prompt=True, enable_thinking=False)
            prompt_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
            answer_tokens = len(tokenizer.encode(ex["reference_text"], add_special_tokens=False))
            lengths.append({"dialog_id": dialog_id, "split": split, "prompt_tokens": prompt_tokens,
                            "answer_tokens": answer_tokens, "tools": len(ex["tools"]),
                            "calls": len(ex["reference"]), "kind": ex["kind"]})
            key = hashlib.sha256(prompt.encode()).hexdigest()
            if prompt_tokens > 2048:
                rejected.append({"id": ex["id"], "reason": "prompt_over_2048"})
                continue
            if key in seen:
                rejected.append({"id": ex["id"], "reason": "duplicate_prompt", "first": seen[key]})
                continue
            seen[key] = ex["id"]
            tool_names.update(t["function"]["name"] for t in ex["tools"])
            ex["prompt"] = prompt
            ex["prompt_tokens"] = prompt_tokens
            rows[split].append(ex)

    train_dialogs = {ex["dialog_id"] for ex in rows["train"]}
    val_dialogs = {ex["dialog_id"] for ex in rows["validation"]}
    assert not train_dialogs & val_dialogs
    rng = random.Random(SEED)
    for split, examples in rows.items():
        rng.shuffle(examples)
        with (data_dir / f"{split}.jsonl").open("w") as f:
            for ex in examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    validation = []
    for kind, n_calls in [("no_call", None), ("call", 1), ("call", 2)]:
        pool = [ex for ex in rows["validation"] if ex["kind"] == kind and
                (n_calls is None or min(len(ex["reference"]), 2) == n_calls)]
        validation.extend(pool[:64])
    save_json(data_dir / "validation_eval.json", validation)
    save_json(ROOT / "results" / "data_examples.json", [rows["train"][i] for i in range(3)])
    save_json(ROOT / "results" / "rejected.json", rejected)
    quantiles = {}
    for col in ["prompt_tokens", "answer_tokens", "tools", "calls"]:
        quantiles[col] = {str(q): float(np.quantile([x[col] for x in lengths], q))
                          for q in [0, .25, .5, .75, .9, .95, .99, 1]}
    manifest = {
        "dataset": "Team-ACE/ToolACE", "dataset_revision": DATA_REVISION,
        "raw_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        "model": MODEL, "model_revision": MODEL_REVISION, "seed": SEED,
        "raw_dialogs": len(raw), "raw_roles": dict(Counter(m["from"] for d in raw for m in d["conversations"])),
        "raw_dialog_turns": dict(Counter(len(d["conversations"]) for d in raw)),
        "prefixes_before_filter": len(lengths), "quantiles_before_filter": quantiles,
        "kinds_before_filter": dict(Counter(x["kind"] for x in lengths)),
        "filters": dict(Counter(x["reason"].split(":")[0] for x in rejected)),
        "unique_tools_kept": len(tool_names), "max_prompt_tokens": 2048,
        "validation_eval_examples": len(validation),
        "splits": {split: {"rows": len(examples), "dialogs": len({x["dialog_id"] for x in examples}),
                            "kinds": dict(Counter(x["kind"] for x in examples)),
                            "sha256": hashlib.sha256((data_dir / f"{split}.jsonl").read_bytes()).hexdigest()}
                   for split, examples in rows.items()}
    }
    save_json(ROOT / "results" / "data_manifest.json", manifest)
    save_json(ROOT / "results" / "eda_lengths.json", lengths)
    save_json(ROOT / "results" / "split_ids.json", {s: [x["id"] for x in ex] for s, ex in rows.items()})
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
