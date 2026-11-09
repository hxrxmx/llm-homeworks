import argparse
import json
from collections import defaultdict
from statistics import mean

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from prepare import MODEL, MODEL_REVISION, ROOT, save_json
from reward import components


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    examples = json.loads((ROOT / "data/validation_eval.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REVISION)
    prompts = [{"prompt_token_ids": tokenizer.encode(ex["prompt"], add_special_tokens=False)} for ex in examples]
    engine = LLM(model=args.model, revision=MODEL_REVISION if args.model == MODEL else None,
                 tokenizer=MODEL, tokenizer_revision=MODEL_REVISION, generation_config="vllm",
                 dtype="bfloat16", language_model_only=True, max_model_len=4096,
                 max_num_seqs=64, gpu_memory_utilization=.85, enforce_eager=True, seed=42)
    outputs = engine.generate(prompts, SamplingParams(temperature=0, max_tokens=1024, skip_special_tokens=False))
    rows = []
    for ex, output in zip(examples, outputs):
        response = output.outputs[0]
        score = components(response.text, ex["tools"], ex["reference"])
        if response.finish_reason == "length":
            score.update(exact=0.0, shaped=0.0, valid=False)
        rows.append({"id": ex["id"], "dialog_id": ex["dialog_id"], "kind": ex["kind"],
                     "reference": ex["reference"], "response": response.text,
                     "finish_reason": response.finish_reason, **score})
    groups = defaultdict(list)
    for row in rows:
        groups[row["kind"]].append(row)
    metrics = {k: {"n": len(group), "exact": mean(r["exact"] for r in group),
                   "shaped": mean(r["shaped"] for r in group)} for k, group in groups.items()}
    metrics["all"] = {"n": len(rows), "exact": mean(r["exact"] for r in rows),
                      "shaped": mean(r["shaped"] for r in rows), "valid": mean(r["valid"] for r in rows)}
    out = ROOT / "results" / args.name
    save_json(out / "validation_metrics.json", metrics)
    with (out / "validation_predictions.jsonl").open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(metrics, indent=2))
