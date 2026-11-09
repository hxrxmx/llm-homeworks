import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoTokenizer

from prepare import MODEL, MODEL_REVISION, ROOT, save_json


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    args = parser.parse_args()
    base = AutoModelForImageTextToText.from_pretrained(MODEL, revision=MODEL_REVISION, dtype=torch.bfloat16)
    adapter = ROOT / "checkpoints" / (args.run + "_adapter")
    model = PeftModel.from_pretrained(base, adapter).merge_and_unload()
    path = ROOT / "checkpoints" / args.run
    model.save_pretrained(path, max_shard_size="4GB")
    AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REVISION).save_pretrained(path)
    save_json(path / "export_complete.json", {"adapter": str(adapter), "model_revision": MODEL_REVISION})
    print("merged checkpoint:", path)
