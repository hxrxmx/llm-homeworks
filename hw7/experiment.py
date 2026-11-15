import argparse
import gc
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from importlib.metadata import version
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig, set_seed

from massive_data import (
    DATASET_REVISION, DEV_SAMPLES, SEED, WIKI_REVISION,
    compute_metrics, load_massive, make_system_prompt, parse_answer, prepare_massive,
)


ROOT = Path(__file__).resolve().parent
ARTIFACTS = Path(os.environ.get("HW7_ARTIFACTS", ROOT / "artifacts_compression_hw"))
BASE_MODEL = "Qwen/Qwen3-1.7B"
BASE_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
TEACHER_MODEL = "Qwen/Qwen3-32B"
TEACHER_REVISION = "9216db5781bf21249d130ec9da846c4624c16137"
MAX_PROMPT_TOKENS = 2048
MAX_NEW_TOKENS = 128
GEN_BATCH = 32
DISTILL_SAMPLES = 2048
CALIB_SAMPLES = 128
CALIB_LENGTH = 640
METRICS = ["quality", "intent_accuracy", "slot_micro_f1", "frame_exact_match", "valid_json_rate"]

QUANT = {
    "awq_massive": {"method": "AWQ", "calibration": "massive", "scheme": "W4A16"},
    "awq_wikitext": {"method": "AWQ", "calibration": "wikitext", "scheme": "W4A16"},
    "gptq_massive": {"method": "GPTQ", "calibration": "massive", "scheme": "W4A16"},
    "gptq_edges_bf16": {"method": "GPTQ", "calibration": "massive", "scheme": "W4A16", "keep_edges": True},
    "smoothquant_massive": {"method": "SmoothQuant", "calibration": "massive",
                          "scheme": "W8A8", "alpha": 0.5},
}
STUDENTS = {
    "sft_512": {"samples": 512, "lr": 2e-5, "epochs": 2},
    "sft_2048": {"samples": 2048, "lr": 2e-5, "epochs": 2},
}


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    tmp.replace(path)


def read_rows(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def clean_memory():
    gc.collect()
    torch.cuda.empty_cache()


def load_model(path, revision=None, quantized=False, training=False):
    kwargs = {"revision": revision, "torch_dtype": torch.float32 if training else torch.bfloat16,
              "attn_implementation": "sdpa"}
    model_class = AutoModelForCausalLM
    if not training:
        kwargs["device_map"] = {"": "cuda:0"}
    if quantized:
        from transformers import CompressedTensorsConfig

        kwargs["quantization_config"] = CompressedTensorsConfig(run_compressed=False)
        config = json.loads((Path(path) / "config.json").read_text())
        if config["quantization_config"]["format"] == "int-quantized":
            from transformers import Qwen3ForCausalLM

            class FrozenQwen(Qwen3ForCausalLM):
                def __init__(self, config):
                    super().__init__(config)
                    # INT8 weights are assigned before decompression in this loader.
                    self.requires_grad_(False)

            model_class = FrozenQwen
    model = model_class.from_pretrained(path, **kwargs)
    return model if training else model.eval()


def encode_prompt(tokenizer, system, text):
    ids = tokenizer.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": text}],
        tokenize=True, add_generation_prompt=True, enable_thinking=False,
    )
    if len(ids) > MAX_PROMPT_TOKENS:
        raise ValueError(f"Промпт имеет {len(ids)} токенов")
    return ids


def trim_answer(ids, eos, pad):
    if eos in ids:
        return ids[:ids.index(eos) + 1]
    while ids and ids[-1] == pad:
        ids.pop()
    return ids


@torch.inference_mode()
def generate_answer_ids(model, tok, prompts):
    eos = tok.eos_token_id
    pad = tok.pad_token_id if tok.pad_token_id is not None else eos
    width = max(map(len, prompts))
    inp = torch.tensor([[pad] * (width - len(ids)) + ids for ids in prompts], device=model.device)
    mask = torch.tensor([[0] * (width - len(ids)) + [1] * len(ids) for ids in prompts], device=model.device)
    config = GenerationConfig(
        do_sample=False, num_beams=1, max_new_tokens=MAX_NEW_TOKENS,
        eos_token_id=eos, pad_token_id=pad, use_cache=True,
    )
    output = model.generate(input_ids=inp, attention_mask=mask, generation_config=config,
                            use_model_defaults=False)
    return [trim_answer(ids, eos, pad) for ids in output[:, width:].tolist()]


def generate_batch(model, tok, system, rows):
    prompts = [encode_prompt(tok, system, row["text"]) for row in rows]
    ids = generate_answer_ids(model, tok, prompts)
    texts = tok.batch_decode(ids, skip_special_tokens=True)
    return texts, prompts, ids


def evaluate(name, model, tok, dev, intents, slot_types, config):
    folder = ARTIFACTS / "eval" / name
    if (folder / "metrics.json").is_file():
        return json.loads((folder / "metrics.json").read_text())
    system = make_system_prompt(intents, slot_types)
    rows = list(dev)
    predictions = []
    started = time.perf_counter()
    for i in tqdm(range(0, len(rows), GEN_BATCH), desc=name):
        batch = rows[i:i + GEN_BATCH]
        texts, _, _ = generate_batch(model, tok, system, batch)
        predictions.extend({**row, "prediction": text} for row, text in zip(batch, texts))
    metrics = compute_metrics([row["prediction"] for row in predictions], dev["gold"], intents, slot_types)
    metrics.update(run=name, config=config, eval_seconds=time.perf_counter() - started)
    write_rows(folder / "predictions.jsonl", predictions)
    write_json(folder / "metrics.json", metrics)
    print(name, {key: round(metrics[key], 4) for key in METRICS}, flush=True)
    return metrics


def collect_teacher_answers(model, tok, train, intents, slot_types):
    target = ARTIFACTS / "teacher_answers" / "answers.jsonl"
    rows = list(train.shuffle(seed=SEED).select(range(DISTILL_SAMPLES)))
    system = make_system_prompt(intents, slot_types)
    records = []
    for i in tqdm(range(0, len(rows), GEN_BATCH), desc="ответы учителя"):
        batch = rows[i:i + GEN_BATCH]
        part = ARTIFACTS / ".work" / "teacher" / f"{i:05d}.jsonl"
        if part.is_file():
            generated = read_rows(part)
            if [row["id"] for row in generated] != [row["id"] for row in batch]:
                raise ValueError(f"Порядок примеров изменился в {part}")
        else:
            texts, prompts, ids = generate_batch(model, tok, system, batch)
            generated = [
                {**row, "teacher_answer": text, "prompt_ids": prompt, "answer_ids": tokens}
                for row, text, prompt, tokens in zip(batch, texts, prompts, ids)
            ]
            write_rows(part, generated)
        records.extend(generated)
    write_rows(target, records)
    valid = sum(parse_answer(row["teacher_answer"], intents, slot_types) is not None for row in records)
    write_json(target.with_name("summary.json"), {
        "n": len(records), "valid": valid,
        "train_ids_sha256": ids_sha(records), "teacher_revision": TEACHER_REVISION,
    })


def ids_sha(rows):
    return hashlib.sha256("\n".join(row["id"] for row in rows).encode()).hexdigest()


def baseline(dev, intents, slot_types):
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, revision=BASE_REVISION)
    model = load_model(BASE_MODEL, BASE_REVISION)
    evaluate("baseline", model, tok, dev, intents, slot_types, {
        "model": BASE_MODEL, "revision": BASE_REVISION, "precision": "bf16",
    })


def teacher(train, dev, intents, slot_types):
    tok = AutoTokenizer.from_pretrained(TEACHER_MODEL, revision=TEACHER_REVISION)
    student_tok = AutoTokenizer.from_pretrained(BASE_MODEL, revision=BASE_REVISION)
    if (tok.get_vocab() != student_tok.get_vocab() or tok.chat_template != student_tok.chat_template
            or tok.eos_token_id != student_tok.eos_token_id or tok.pad_token_id != student_tok.pad_token_id):
        raise ValueError("Токенизаторы учителя и ученика отличаются")
    model = load_model(TEACHER_MODEL, TEACHER_REVISION)
    if model.config.vocab_size != AutoConfig.from_pretrained(BASE_MODEL, revision=BASE_REVISION).vocab_size:
        raise ValueError("Размеры словарей моделей отличаются")
    evaluate("teacher", model, tok, dev, intents, slot_types, {
        "model": TEACHER_MODEL, "revision": TEACHER_REVISION, "precision": "bf16",
    })
    if not (ARTIFACTS / "teacher_answers" / "summary.json").is_file():
        collect_teacher_answers(model, tok, train, intents, slot_types)


def calibration_data(source, train, tok, system):
    from datasets import Dataset, load_dataset

    selected = train.shuffle(seed=SEED).select(range(CALIB_SAMPLES))
    texts = list(selected["text"])
    ids = selected["id"]
    if source == "massive":
        snippets = texts
    else:
        refs = tok(texts, add_special_tokens=False).input_ids
        file = (f"hf://datasets/Salesforce/wikitext@{WIKI_REVISION}/"
                "wikitext-2-raw-v1/train-00000-of-00001.parquet")
        wiki = load_dataset("parquet", data_files={"train": file})["train"].shuffle(seed=SEED)
        chosen = []
        for i, row in enumerate(wiki):
            text = row["text"].strip()
            if len(text.split()) < 8 or text.startswith("="):
                continue
            n = len(refs[len(chosen)])
            tokens = tok(text, add_special_tokens=False).input_ids
            if len(tokens) >= n:
                chosen.append((str(i), tok.decode(tokens[:n])))
            if len(chosen) == CALIB_SAMPLES:
                break
        ids, snippets = zip(*chosen)
    prompts = [encode_prompt(tok, system, text) for text in snippets]
    if max(map(len, prompts)) > CALIB_LENGTH:
        raise ValueError("Увеличьте лимит калибровочного промпта")
    data = Dataset.from_dict({"input_ids": prompts, "attention_mask": [[1] * len(ids) for ids in prompts]})
    return data, {
        "calib_samples": len(prompts), "calib_tokens": sum(map(len, prompts)),
        "calib_max_length": CALIB_LENGTH, "calib_ids": list(ids),
    }


def checkpoint_bytes(path):
    return sum(file.stat().st_size for file in path.glob("*.safetensors"))


def quantize(name, train, dev, intents, slot_types):
    from llmcompressor import oneshot
    from llmcompressor.modifiers.awq import AWQModifier
    from llmcompressor.modifiers.quantization import GPTQModifier, QuantizationModifier
    from llmcompressor.modifiers.smoothquant import SmoothQuantModifier

    params = QUANT[name]
    path = ARTIFACTS / "quantized" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.is_dir():
        tok = AutoTokenizer.from_pretrained(BASE_MODEL, revision=BASE_REVISION)
        data, calib = calibration_data(params["calibration"], train, tok, make_system_prompt(intents, slot_types))
        model = load_model(BASE_MODEL, BASE_REVISION)
        ignored = ["lm_head"]
        kept = []
        if params.get("keep_edges"):
            kept = [0, model.config.num_hidden_layers - 1]
            ignored.extend(fr"re:^model\.layers\.{layer}\..*" for layer in kept)
        opts = dict(targets=["Linear"], scheme=params["scheme"], ignore=ignored)
        if params["method"] == "AWQ":
            recipe = AWQModifier(**opts)
        elif params["method"] == "GPTQ":
            recipe = GPTQModifier(**opts)
        else:
            recipe = [SmoothQuantModifier(smoothing_strength=params["alpha"]), QuantizationModifier(**opts)]
        started = time.perf_counter()
        with tempfile.TemporaryDirectory(dir=path.parent) as tmp:
            checkpoint = Path(tmp) / "model"
            oneshot(
                model=model, processor=tok, dataset=data, recipe=recipe,
                num_calibration_samples=CALIB_SAMPLES, max_seq_length=CALIB_LENGTH,
                shuffle_calibration_samples=False, pad_to_max_length=False,
                output_dir=str(checkpoint), save_compressed=True,
                log_dir=str(ARTIFACTS / ".work" / "quant_logs"),
            )
            size = checkpoint_bytes(checkpoint)
            quant_config = json.loads((checkpoint / "config.json").read_text()).get("quantization_config")
            if not size or not quant_config:
                raise RuntimeError(f"{name}: файл с квантизированными весами не сохранен")
            write_json(checkpoint / "run.json", {
                **params, **calib, "kept_layers": kept, "base_revision": BASE_REVISION,
                "bytes": size, "quantize_seconds": time.perf_counter() - started,
            })
            checkpoint.rename(path)
        del model
        clean_memory()

    config = json.loads((path / "run.json").read_text())
    tok = AutoTokenizer.from_pretrained(path)
    model = load_model(path, quantized=True)
    evaluate(name, model, tok, dev, intents, slot_types, config)


class TeacherAnswerDataset(torch.utils.data.Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        prompt, answer = row["prompt_ids"], row["answer_ids"]
        return {
            "input_ids": prompt + answer,
            "attention_mask": [1] * (len(prompt) + len(answer)),
            "labels": [-100] * len(prompt) + answer,
        }


def collate_teacher_answers(rows, pad_id):
    width = max(len(row["input_ids"]) for row in rows)
    padding = {"input_ids": pad_id, "attention_mask": 0, "labels": -100}
    return {
        key: torch.tensor([row[key] + [pad] * (width - len(row[key])) for row in rows])
        for key, pad in padding.items()
    }


def train_student(name, dev, intents, slot_types):
    from transformers import Trainer, TrainingArguments

    params = STUDENTS[name]
    path = ARTIFACTS / "student_candidates" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, revision=BASE_REVISION)
    if not path.is_dir():
        rows = read_rows(ARTIFACTS / "teacher_answers" / "answers.jsonl")[:params["samples"]]
        valid = [row for row in rows if row["answer_ids"] and parse_answer(row["teacher_answer"], intents, slot_types)]
        if not valid:
            raise ValueError("Нет валидных ответов учителя")
        model = load_model(BASE_MODEL, BASE_REVISION, training=True)
        model.config.use_cache = False
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        with tempfile.TemporaryDirectory(dir=path.parent) as tmp:
            checkpoint = Path(tmp) / "model"
            args = TrainingArguments(
                output_dir=str(Path(tmp) / "trainer"), bf16=True, tf32=True,
                per_device_train_batch_size=4, gradient_accumulation_steps=4,
                num_train_epochs=params["epochs"], learning_rate=params["lr"],
                lr_scheduler_type="cosine", warmup_ratio=0.05, weight_decay=0.01,
                optim="adamw_torch_fused", gradient_checkpointing=True,
                gradient_checkpointing_kwargs={"use_reentrant": False},
                logging_steps=10, save_strategy="no", report_to="none",
                seed=SEED, data_seed=SEED,
            )
            trainer = Trainer(
                model=model, args=args, train_dataset=TeacherAnswerDataset(valid),
                data_collator=lambda batch: collate_teacher_answers(batch, pad_id),
            )
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            trainer.train()
            seconds = time.perf_counter() - started
            peak = torch.cuda.max_memory_allocated() / 2**30
            history = trainer.state.log_history
            steps = trainer.state.global_step
            del trainer
            model.config.use_cache = True
            model.to(torch.bfloat16).save_pretrained(checkpoint, max_shard_size="1500MB")
            tok.save_pretrained(checkpoint)
            write_json(checkpoint / "run.json", {
                **params, "valid_samples": len(valid), "train_ids_sha256": ids_sha(valid),
                "batch_size": 4, "grad_accum": 4, "steps": steps, "base_revision": BASE_REVISION,
                "train_seconds": seconds, "peak_gib": peak, "bytes": checkpoint_bytes(checkpoint),
            })
            write_json(ARTIFACTS / "training" / f"{name}.json", history)
            checkpoint.rename(path)
        del model
        clean_memory()

    model = load_model(path)
    config = json.loads((path / "run.json").read_text())
    evaluate(name, model, tok, dev, intents, slot_types, config)


def select_student():
    scores = {
        name: json.loads((ARTIFACTS / "eval" / name / "metrics.json").read_text())
        for name in STUDENTS
    }
    best = max(scores, key=lambda name: scores[name]["quality"])
    target = ARTIFACTS / "student_sft"
    if not target.is_dir():
        with tempfile.TemporaryDirectory(dir=ARTIFACTS) as tmp:
            copy = Path(tmp) / "model"
            shutil.copytree(ARTIFACTS / "student_candidates" / best, copy)
            copy.rename(target)
    write_json(ARTIFACTS / "student_selection.json", {"run": best, "quality": scores[best]["quality"]})
    print(f"student_sft: {best}, Q={scores[best]['quality']:.4f}", flush=True)


def done(name):
    evaluated = (ARTIFACTS / "eval" / name / "metrics.json").is_file()
    if name == "teacher":
        return evaluated and (ARTIFACTS / "teacher_answers" / "summary.json").is_file()
    return evaluated


def run(name):
    if done(name):
        print(f"{name} уже оценен", flush=True)
        return
    set_seed(SEED)
    train, dev, intents, slot_types = prepare_massive(load_massive())
    write_json(ARTIFACTS / "data_manifest.json", {
        "dataset_revision": DATASET_REVISION, "seed": SEED,
        "train": len(train), "dev": DEV_SAMPLES, "dev_ids": list(dev["id"]), "dev_ids_sha256": ids_sha(dev),
        "intents": intents, "slot_types": slot_types,
    })
    write_json(ARTIFACTS / "environment.json", {
        **{p: version(p) for p in ("torch", "transformers", "datasets", "llmcompressor", "compressed-tensors",
                                  "accelerate", "safetensors", "huggingface-hub")},
        "gpu": torch.cuda.get_device_name(),
        "cuda": torch.version.cuda,
        "generate_batch_size": GEN_BATCH,
        "max_new_tokens": MAX_NEW_TOKENS,
    })
    if name == "baseline":
        baseline(dev, intents, slot_types)
    elif name == "teacher":
        teacher(train, dev, intents, slot_types)
    elif name in QUANT:
        quantize(name, train, dev, intents, slot_types)
    else:
        train_student(name, dev, intents, slot_types)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    jobs = ["baseline", "teacher", *STUDENTS, *QUANT]
    parser.add_argument("job", choices=["all", "select_student", *jobs])
    job = parser.parse_args().job
    if job == "all":
        for name in jobs:
            subprocess.run([sys.executable, str(ROOT / "experiment.py"), name], check=True)
        select_student()
    elif job == "select_student":
        select_student()
    else:
        run(job)
