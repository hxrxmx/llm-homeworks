import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import Dataset, load_dataset, load_from_disk
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed


ROOT = Path(__file__).resolve().parent
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DATASET = "fancyzhx/ag_news"
DATASET_REVISION = "eb185aade064a813bc0b7f42de02595523103ca4"
MODEL_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
SEED = 42
ARTICLE_TOKENS = 256
CLASSES = ["World", "Sports", "Business", "Sci/Tech"]
LORA_BASE = {"r": 16, "lora_alpha": 32, "lora_dropout": 0.05, "target_modules": ["q_proj", "v_proj"]}
RUNS = {
    "base": {"mode": "base"},
    "full_ft": {"mode": "full", "lr": 2e-5},
    "lora_base": {"mode": "lora", "lr": 2e-4, **LORA_BASE},
    "lora_r64": {"mode": "lora", "lr": 2e-4, **LORA_BASE, "r": 64},
    "lora_alpha64": {"mode": "lora", "lr": 2e-4, **LORA_BASE, "lora_alpha": 64},
    "lora_dropout15": {"mode": "lora", "lr": 2e-4, **LORA_BASE, "lora_dropout": 0.15},
    "lora_all_linear": {"mode": "lora", "lr": 2e-4, **LORA_BASE, "target_modules": "all-linear"},
}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def read_json(path):
    return json.loads(path.read_text())


def text_key(text):
    return " ".join(text.casefold().split())


def prompt_messages(text):
    return [
        {"role": "system", "content": "You classify news articles by topic."},
        {"role": "user", "content": (
            "Choose exactly one category for the news article.\n"
            "1 = World\n2 = Sports\n3 = Business\n4 = Science and Technology\n\n"
            f"Article:\n{text}\n\nReply with one digit (1, 2, 3, or 4) and nothing else."
        )},
    ]


def encode_rows(rows, tokenizer, train=False):
    article_ids = tokenizer([row["text"] for row in rows], add_special_tokens=False)["input_ids"]
    prompts = []
    for row, ids in zip(rows, article_ids):
        text = row["text"] if len(ids) <= ARTICLE_TOKENS else tokenizer.decode(ids[:ARTICLE_TOKENS])
        prompts.append(tokenizer.apply_chat_template(prompt_messages(text), tokenize=False, add_generation_prompt=True))
    inputs = tokenizer(prompts, add_special_tokens=False)["input_ids"]
    if train:
        inputs = [ids + tokenizer.encode(str(row["label"] + 1), add_special_tokens=False) + [tokenizer.eos_token_id]
                  for ids, row in zip(inputs, rows)]
    data = Dataset.from_dict({"input_ids": inputs})
    stats = {"rows": len(rows), "article_truncated": sum(len(ids) > ARTICLE_TOKENS for ids in article_ids),
             "max_sequence": max(map(len, inputs)), "mean_sequence": float(np.mean(list(map(len, inputs))))}
    return data, stats


def prepare():
    out = ROOT / "outputs"
    if (out / "data_manifest.json").exists() and all((ROOT / "data" / name / "dataset_info.json").exists()
                                                       for name in ["train", "validation", "test"]):
        print("data already prepared", flush=True)
        return
    dataset_revision = DATASET_REVISION
    model_revision = MODEL_REVISION
    raw = load_dataset(DATASET, revision=dataset_revision)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=model_revision)
    label_ids = [tokenizer.encode(str(i), add_special_tokens=False) for i in range(1, 5)]
    assert all(len(ids) == 1 for ids in label_ids)

    test_rows = [{"id": i, **row} for i, row in enumerate(raw["test"])]
    test_keys = {text_key(row["text"]) for row in test_rows}
    seen = set()
    pool = [[] for _ in CLASSES]
    duplicates, overlaps = 0, 0
    for i, row in enumerate(raw["train"]):
        key = text_key(row["text"])
        if key in test_keys:
            overlaps += 1
            continue
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        pool[row["label"]].append({"id": i, **row})

    rng = np.random.default_rng(SEED)
    train_rows, val_rows = [], []
    for rows in pool:
        indices = rng.permutation(len(rows))
        val_rows.extend(rows[i] for i in indices[:500])
        train_rows.extend(rows[i] for i in indices[500:5500])
    rng.shuffle(train_rows)
    rng.shuffle(val_rows)
    assert len(train_rows) == 20000 and len(val_rows) == 2000
    assert not ({text_key(r["text"]) for r in train_rows} & {text_key(r["text"]) for r in val_rows})

    stats = {}
    for name, rows in [("train", train_rows), ("validation", val_rows), ("test", test_rows)]:
        encoded, stats[name] = encode_rows(rows, tokenizer, train=name == "train")
        encoded.save_to_disk(str(ROOT / "data" / name))
        pd.DataFrame(rows).to_json(ROOT / "data" / f"{name}_rows.jsonl", orient="records", lines=True, force_ascii=False)
    manual = []
    for label, count in enumerate([3, 3, 2, 2]):
        manual.extend([row for row in test_rows if row["label"] == label][:count])
    write_json(out / "manual_examples.json", manual)
    write_json(out / "train_examples.json", [next(row for row in train_rows if row["label"] == label)
                                             for label in range(4)])
    split_ids = {"train": [r["id"] for r in train_rows], "validation": [r["id"] for r in val_rows],
                 "test": [r["id"] for r in test_rows]}
    write_json(out / "split_ids.json", split_ids)
    manifest = {
        "dataset": DATASET, "dataset_revision": dataset_revision, "model": MODEL,
        "model_revision": model_revision, "seed": SEED, "raw_train": len(raw["train"]),
        "raw_test": len(raw["test"]), "removed_train_test_overlap": overlaps,
        "removed_train_duplicates": duplicates, "article_token_limit": ARTICLE_TOKENS,
        "stats": stats, "train_class_counts": dict(Counter(r["label"] for r in train_rows)),
        "validation_class_counts": dict(Counter(r["label"] for r in val_rows)),
        "split_sha256": hashlib.sha256(json.dumps(split_ids, sort_keys=True).encode()).hexdigest(),
    }
    write_json(out / "data_manifest.json", manifest)
    print(json.dumps(manifest, indent=2), flush=True)


class AnswerTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs, use_cache=False, logits_to_keep=3)
        loss = F.cross_entropy(outputs.logits[:, :-1].float().reshape(-1, outputs.logits.shape[-1]), labels.reshape(-1))
        return (loss, outputs) if return_outputs else loss


def pad_batch(rows, tokenizer, train=False):
    batch = tokenizer.pad(rows, padding=True, pad_to_multiple_of=8, return_tensors="pt")
    if train:
        batch["labels"] = batch["input_ids"][:, -2:].clone()
    return batch


def evaluate(model, tokenizer, split):
    model.eval()
    data = load_from_disk(str(ROOT / "data" / split))
    rows = pd.read_json(ROOT / "data" / f"{split}_rows.jsonl", lines=True)
    label_tokens = {tokenizer.encode(str(i + 1), add_special_tokens=False)[0]: i for i in range(4)}
    loader = DataLoader(data, batch_size=64, collate_fn=lambda batch: pad_batch(batch, tokenizer))
    predictions, tokens, nll = [], [], []
    offset = 0
    with torch.inference_mode():
        for batch in loader:
            batch = batch.to(model.device)
            logits = model(**batch, use_cache=False, logits_to_keep=1).logits[:, -1].float()
            ids = logits.argmax(-1).tolist()
            predictions.extend(label_tokens.get(token, -1) for token in ids)
            tokens.extend(tokenizer.decode([token]) for token in ids)
            targets = torch.tensor([tokenizer.encode(str(int(label) + 1), add_special_tokens=False)[0]
                                    for label in rows["label"].iloc[offset:offset + len(ids)]], device=model.device)
            nll.extend(F.cross_entropy(logits, targets, reduction="none").tolist())
            offset += len(ids)
    rows["prediction"] = predictions
    rows["first_token"] = tokens
    rows["label_nll"] = nll
    metrics = {
        "accuracy": accuracy_score(rows["label"], predictions),
        "macro_f1": f1_score(rows["label"], predictions, labels=list(range(4)), average="macro", zero_division=0),
        "invalid_rate": float(np.mean(np.array(predictions) < 0)),
        "label_nll": float(np.mean(nll)),
        "per_class": classification_report(rows["label"], predictions, labels=list(range(4)),
                                            target_names=CLASSES, output_dict=True, zero_division=0),
    }
    return metrics, rows


def manual_outputs(model, tokenizer):
    rows = read_json(ROOT / "outputs" / "manual_examples.json")
    encoded, _ = encode_rows(rows, tokenizer)
    answers = []
    for row, item in zip(rows, encoded):
        batch = pad_batch([item], tokenizer).to(model.device)
        with torch.inference_mode():
            output = model.generate(**batch, max_new_tokens=12, do_sample=False,
                                    pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        answer = tokenizer.decode(output[0, batch.input_ids.shape[1]:], skip_special_tokens=True).strip()
        answers.append({**row, "expected": str(row["label"] + 1), "answer": answer})
    return answers


def run(name, max_steps=-1):
    config = RUNS[name]
    folder = ROOT / "outputs" / (name if max_steps < 0 else f"smoke_{name}")
    if (folder / "summary.json").exists():
        print(name, "already finished", flush=True)
        return
    folder.mkdir(parents=True, exist_ok=True)
    manifest = read_json(ROOT / "outputs" / "data_manifest.json")
    set_seed(SEED)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=manifest["model_revision"], padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=manifest["model_revision"],
                                                torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda()
    model.config.pad_token_id = tokenizer.pad_token_id
    if config["mode"] == "lora":
        lora = {key: config[key] for key in LORA_BASE}
        model = get_peft_model(model, LoraConfig(task_type=TaskType.CAUSAL_LM, bias="none", **lora))
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) if name != "base" else 0
    summary = {"run": name, "config": config, "seed": SEED, "total_parameters": total,
               "trainable_parameters": trainable, "trainable_percent": 100 * trainable / total,
               "gpu": torch.cuda.get_device_name(), "gpu_total_gib": torch.cuda.get_device_properties(0).total_memory / 2**30,
               "versions": {lib: importlib.metadata.version(lib) for lib in ["torch", "transformers", "peft", "datasets", "accelerate"]},
               "split_sha256": manifest["split_sha256"], "model_revision": manifest["model_revision"]}

    if name != "base":
        args = TrainingArguments(
            output_dir=str(ROOT / "checkpoints" / (name if max_steps < 0 else f"smoke_{name}")),
            num_train_epochs=1, max_steps=max_steps, learning_rate=config["lr"],
            per_device_train_batch_size=16, gradient_accumulation_steps=2,
            optim="adamw_torch_fused", lr_scheduler_type="cosine", warmup_ratio=0.05,
            weight_decay=0.01, max_grad_norm=1.0, bf16=True, tf32=True,
            save_strategy="no", eval_strategy="no", logging_steps=50, logging_first_step=True,
            report_to="none", dataloader_num_workers=0, seed=SEED, data_seed=SEED,
            disable_tqdm=True, remove_unused_columns=False,
        )
        trainer = AnswerTrainer(model=model, args=args, processing_class=tokenizer,
                                train_dataset=load_from_disk(str(ROOT / "data" / "train")),
                                data_collator=lambda batch: pad_batch(batch, tokenizer, train=True))
        trainer.model_accepts_loss_kwargs = False
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        result = trainer.train()
        torch.cuda.synchronize()
        summary.update({
            "train_seconds": time.perf_counter() - started,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
            "optimizer_steps": trainer.state.global_step, "train_metrics": result.metrics,
            "training_args": args.to_dict(),
        })
        write_json(folder / "log_history.json", trainer.state.log_history)
        if max_steps < 0:
            trainer.save_model()
        if max_steps > 0:
            write_json(folder / "summary.json", summary)
            return

    for split in ["validation", "test"]:
        metrics, predictions = evaluate(model, tokenizer, split)
        summary[split] = metrics
        predictions.drop(columns="text").to_csv(folder / f"{split}_predictions.csv", index=False)
    write_json(folder / "manual.json", manual_outputs(model, tokenizer))
    write_json(folder / "summary.json", summary)
    print(json.dumps({"run": name, "test": summary["test"], "train_seconds": summary.get("train_seconds")}, indent=2), flush=True)


def self_check():
    from transformers import Qwen2Config, Qwen2ForCausalLM

    set_seed(1)
    model = Qwen2ForCausalLM(Qwen2Config(vocab_size=32, hidden_size=16, intermediate_size=32,
                                       num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1)).eval()
    ids = torch.tensor([[0, 0, 3, 4, 5, 6], [1, 2, 3, 4, 7, 6]])
    mask = (ids != 0).long()
    labels = ids.clone()
    labels[:, :-2] = -100
    expected = model(input_ids=ids, attention_mask=mask, labels=labels).loss
    logits = model(input_ids=ids, attention_mask=mask, logits_to_keep=3).logits
    actual = F.cross_entropy(logits[:, :-1].reshape(-1, 32), ids[:, -2:].reshape(-1))
    assert torch.allclose(expected, actual, atol=1e-6), (expected, actual)
    assert set(RUNS["lora_base"]) == set(RUNS["lora_r64"])
    for name, key in [("lora_r64", "r"), ("lora_alpha64", "lora_alpha"),
                      ("lora_dropout15", "lora_dropout"), ("lora_all_linear", "target_modules")]:
        assert [k for k in RUNS["lora_base"] if RUNS[name][k] != RUNS["lora_base"][k]] == [key]
    print("loss equals standard masked causal LM loss; four one-factor ablations are valid", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare", "all", "check", *RUNS])
    parser.add_argument("--max-steps", type=int, default=-1)
    args = parser.parse_args()
    if args.command == "check":
        self_check()
    elif args.command == "prepare":
        prepare()
    elif args.command == "all":
        prepare()
        for name in RUNS:
            subprocess.run([sys.executable, "-u", __file__, name], check=True)
    else:
        run(args.command, args.max_steps)
