import argparse
import importlib.metadata
import json
import os
import time
from pathlib import Path

os.environ.setdefault("HF_HOME", "/opt/debug-workspace/hw5-cache/huggingface")
os.environ.setdefault("VLLM_CACHE_ROOT", "/opt/debug-workspace/hw5-cache/vllm")
os.environ.setdefault("TRITON_CACHE_DIR", "/opt/debug-workspace/hw5-cache/triton")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from datasets import Dataset
from huggingface_hub import snapshot_download
from peft import LoraConfig
from transformers import AutoTokenizer, TrainerCallback
from transformers.trainer_utils import get_last_checkpoint
from trl import GRPOConfig, GRPOTrainer

from prepare import MODEL, MODEL_REVISION, ROOT, SEED, save_json
from reward import make_reward


RUNS = {
    "grpo_exact": {"reward": "exact", "loss_type": "grpo", "scale_rewards": "group"},
    "grpo_shaped": {"reward": "shaped", "loss_type": "grpo", "scale_rewards": "group"},
    "dr_grpo_shaped": {"reward": "shaped", "loss_type": "dr_grpo", "scale_rewards": "none"},
}


class Log(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        with (Path(args.output_dir) / "history.jsonl").open("a") as f:
            f.write(json.dumps({"step": state.global_step, **(logs or {})}) + "\n")


def train(name, steps, smoke=False):
    cfg = RUNS[name]
    batch = 8
    run_name = f"smoke_bs{batch}_awake" if smoke else name
    out = ROOT / "runs" / run_name
    out.mkdir(parents=True, exist_ok=True)
    if (out / "summary.json").exists():
        raise FileExistsError(f"run already finished: {out}")
    rows = [json.loads(line) for line in (ROOT / "data/train.jsonl").read_text().splitlines()]
    dataset = Dataset.from_list([{
        "prompt": row["prompt"], "tools_json": json.dumps(row["tools"], ensure_ascii=False),
        "reference_json": json.dumps(row["reference"], ensure_ascii=False), "example_id": row["id"]
    } for row in rows])
    model_path = snapshot_download(MODEL, revision=MODEL_REVISION)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REVISION, padding_side="left")
    torch.backends.cuda.matmul.allow_tf32 = True
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
                                      "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj"])
    args = GRPOConfig(
        output_dir=str(out), seed=SEED, data_seed=SEED,
        model_init_kwargs={"dtype": "bfloat16", "attn_implementation": "sdpa"},
        bf16=True, tf32=True, learning_rate=2e-5, lr_scheduler_type="constant_with_warmup",
        warmup_steps=10 if not smoke else 0, weight_decay=.01, max_grad_norm=.3,
        per_device_train_batch_size=batch, gradient_accumulation_steps=32 // batch,
        num_generations=4, num_iterations=1, max_steps=steps,
        max_completion_length=512, temperature=.8, top_p=.95, top_k=0,
        beta=.02, epsilon=.2, loss_type=cfg["loss_type"], scale_rewards=cfg["scale_rewards"],
        mask_truncated_completions=True, disable_dropout=True,
        use_vllm=True, vllm_mode="colocate", vllm_gpu_memory_utilization=.35,
        vllm_max_model_length=2560, vllm_enable_sleep_mode=False,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch_fused", logging_steps=1, report_to="none",
        save_strategy="steps", save_steps=100, save_total_limit=3,
        log_completions=False, dataloader_num_workers=0,
    )
    save_json(out / "config.json", {"name": name, **cfg, "training": args.to_dict(), "lora": {
        "r": lora.r, "alpha": lora.lora_alpha, "dropout": lora.lora_dropout,
        "target_modules": sorted(lora.target_modules)
    }})
    start = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    trainer = GRPOTrainer(model=model_path, args=args, train_dataset=dataset,
                          processing_class=tokenizer, reward_funcs=make_reward(cfg["reward"]),
                          peft_config=lora, callbacks=[Log()])
    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in trainer.model.parameters())
    checkpoint = get_last_checkpoint(str(out))
    trainer.train(resume_from_checkpoint=checkpoint)
    adapter_dir = ROOT / "checkpoints" / (run_name + "_adapter")
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(adapter_dir)
    trainer.save_state()
    save_json(out / "summary.json", {
        "name": name, "selected_step": trainer.state.global_step, "steps_planned": steps,
        "resumed_from": checkpoint,
        "elapsed_seconds": time.monotonic() - start, "trainable_parameters": trainable,
        "total_parameters": total, "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "gpu": torch.cuda.get_device_name(), "model_revision": MODEL_REVISION,
        "data_manifest": json.loads((ROOT / "results/data_manifest.json").read_text()),
        "versions": {p: importlib.metadata.version(p) for p in
                     ["torch", "transformers", "trl", "peft", "vllm", "datasets", "flash-linear-attention"]}
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run", choices=list(RUNS))
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--smoke", action="store_true")
    opts = parser.parse_args()
    train(opts.run, opts.steps, opts.smoke)
