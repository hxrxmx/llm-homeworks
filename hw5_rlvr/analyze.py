import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from prepare import ROOT, save_json


names = ["base", "grpo_exact", "grpo_shaped", "dr_grpo_shaped"]
results = ROOT / "results"
figures = results / "figures"
figures.mkdir(exist_ok=True)

table = []
predictions = {}
for name in names:
    metrics = json.loads((results / name / "metrics.json").read_text())
    table.append({"run": name, "overall": metrics["overall"], **metrics["groups"]})
    predictions[name] = pd.read_json(results / name / "predictions.jsonl", lines=True).set_index("id")
    if name != "base":
        assert predictions[name].index.equals(predictions["base"].index)
table = pd.DataFrame(table).set_index("run")
table.to_csv(results / "comparison.csv")

error_counts = []
for name, pred in predictions.items():
    counts = {"truncated": 0, "empty": 0, "extra_call_irrelevance": 0, "missing_call": 0, "other": 0}
    for _, row in pred.iterrows():
        if row["check"]["valid"]:
            continue
        if row["finish_reason"] == "length":
            kind = "truncated"
        elif not row["response"].strip():
            kind = "empty"
        elif "irrelevance" in row["category"] and row["calls"]:
            kind = "extra_call_irrelevance"
        elif "irrelevance" not in row["category"] and not row["calls"]:
            kind = "missing_call"
        else:
            kind = "other"
        counts[kind] += 1
    error_counts.append({"run": name, **counts})
pd.DataFrame(error_counts).set_index("run").to_csv(results / "error_counts.csv")

lengths = pd.read_json(results / "eda_lengths.json")
fig, axes = plt.subplots(1, 2, figsize=(10, 3))
axes[0].hist(lengths["prompt_tokens"], bins=45, color="#4477aa")
axes[0].axvline(2048, color="#cc6677", linestyle="--", label="filter: 2048")
axes[0].set(xlabel="prompt tokens", ylabel="examples")
axes[0].legend()
counts = lengths["calls"].value_counts().sort_index()
axes[1].bar(counts.index, counts.values, color="#228833")
axes[1].set(xlabel="reference calls", ylabel="examples", xticks=counts.index)
fig.tight_layout()
fig.savefig(figures / "eda.png", dpi=180)
plt.close(fig)

fig, axes = plt.subplots(3, 2, figsize=(10, 8))
resource_rows = []
for i, name in enumerate(names[1:]):
    history = pd.read_json(ROOT / "runs" / name / "history.jsonl", lines=True)
    history = history.dropna(subset=["loss"]).drop_duplicates("step", keep="last").sort_values("step")
    for j, col in enumerate(["loss", "reward"]):
        data = history.dropna(subset=[col])
        axes[i, j].plot(data["step"], data[col], alpha=.25, linewidth=.7)
        axes[i, j].plot(data["step"], data[col].rolling(10, min_periods=1).mean(), linewidth=1.3)
        axes[i, j].set(xlabel="optimizer step", ylabel=col, title=name)
    summary = json.loads((ROOT / "runs" / name / "summary.json").read_text())
    resource_rows.append({"run": name, "steps": summary["selected_step"],
                          "seconds": summary["elapsed_seconds"],
                          "trainable_parameters": summary["trainable_parameters"],
                          "peak_allocated_gib": summary["peak_allocated_gib"],
                          "peak_reserved_gib": summary["peak_reserved_gib"],
                          "first20_reward": history["reward"].head(20).mean(),
                          "last20_reward": history["reward"].tail(20).mean(),
                          "mean_zero_std": history["frac_reward_zero_std"].mean(),
                          "mean_kl": history["kl"].mean(),
                          "mean_clipped_ratio": history["completions/clipped_ratio"].mean(),
                          "mean_completion_tokens": history["completions/mean_length"].mean(),
                          "mean_step_seconds": history["step_time"].mean()})
fig.tight_layout()
fig.savefig(figures / "training.png", dpi=180)
plt.close(fig)
pd.DataFrame(resource_rows).set_index("run").to_csv(results / "resources.csv")

categories = predictions["base"]["category"].to_numpy()
correct = {name: np.array([x["valid"] for x in pred["check"]], dtype=float)
           for name, pred in predictions.items()}
strata = [np.flatnonzero(categories == c) for c in sorted(set(categories))]
weights = {"simple": .2, "multiple": .1, "live_multiple": .1, "parallel": .2,
           "parallel_multiple": .2, "irrelevance": .1, "live_irrelevance": .1}
sample_weight = np.array([weights[c] for c in sorted(set(categories))])
rng = np.random.default_rng(42)
bootstrap = []
pairs = [(name, "base") for name in names[1:]] + [
    ("grpo_shaped", "grpo_exact"), ("grpo_shaped", "dr_grpo_shaped")
]
for name, reference in pairs:
    diff = correct[name] - correct[reference]
    scores = []
    for _ in range(3000):
        means = [diff[rng.choice(ids, len(ids), replace=True)].mean() for ids in strata]
        scores.append(np.dot(sample_weight, means))
    low, high = np.quantile(scores, [.025, .975])
    bootstrap.append({"run": name, "reference": reference,
                      "delta_overall": table.loc[name, "overall"] - table.loc[reference, "overall"],
                      "low": float(low), "high": float(high)})
save_json(results / "bootstrap.json", bootstrap)
print((table * 100).round(2).to_string())
print(pd.DataFrame(bootstrap).to_string(index=False))
