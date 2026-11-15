import json
import re
from collections import Counter


SEED = 42
DATASET_ID = "AmazonScience/massive"
DATASET_REVISION = "ed58ac423a2f4121720918bf5301577edce4ffd3"
DEV_SAMPLES = 256
WIKI_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
SLOT_PATTERN = re.compile(r"\[([^:\[\]]+?)\s*:\s*([^\[\]]*?)\]")


def load_massive():
    from datasets import load_dataset

    url = f"hf://datasets/{DATASET_ID}@{DATASET_REVISION}/ru-RU"
    files = {split: f"{url}/{split}/0000.parquet" for split in ("train", "validation", "test")}
    return load_dataset("parquet", data_files=files)


def extract_slots(annot_utt):
    return [[m[1].strip(), m[2].strip()] for m in SLOT_PATTERN.finditer(annot_utt)]


def prepare_massive(raw):
    feature = raw["train"].features["intent"]
    intents = list(feature.names)
    slot_types = sorted({kind for text in raw["train"]["annot_utt"] for kind, _ in extract_slots(text)})

    def convert(row):
        gold = {"intent": feature.int2str(row["intent"]), "slots": extract_slots(row["annot_utt"])}
        return {
            "id": str(row["id"]),
            "text": row["utt"],
            "gold": json.dumps(gold, ensure_ascii=False, separators=(",", ":")),
        }

    train = raw["train"].map(convert, remove_columns=raw["train"].column_names)
    dev = raw["validation"].shuffle(seed=SEED).select(range(DEV_SAMPLES))
    dev = dev.map(convert, remove_columns=dev.column_names)
    return train, dev, intents, slot_types


def make_system_prompt(intents, slot_types):
    return (
        'Определи intent и извлеки slots. Верни только JSON вида '
        '{"intent":"...","slots":[["тип","значение"],...]}. '
        'Значения slots копируй из запроса. Если slots нет, верни пустой список. '
        'Не добавляй объяснения.\nДопустимые intents: ' + ', '.join(intents)
        + '\nДопустимые типы slots: ' + ', '.join(slot_types)
    )


def parse_answer(text, intents, slot_types):
    try:
        obj = json.loads(text.strip())
        if not isinstance(obj, dict) or set(obj) != {"intent", "slots"}:
            return None
        if not isinstance(obj["intent"], str) or obj["intent"] not in intents:
            return None
        if not isinstance(obj["slots"], list):
            return None
        for slot in obj["slots"]:
            if not isinstance(slot, list) or len(slot) != 2:
                return None
            kind, value = slot
            if (not isinstance(kind, str) or kind not in slot_types
                    or not isinstance(value, str) or not value.strip()):
                return None
        return obj
    except (ValueError, TypeError):
        return None


def slot_counter(obj):
    return Counter((kind, " ".join(value.casefold().split())) for kind, value in obj["slots"])


def compute_metrics(predictions, golds, intents, slot_types):
    if len(predictions) != len(golds) or not golds:
        raise ValueError("Нужны непустые списки одинаковой длины.")
    correct_intent = exact = valid = tp = fp = fn = 0
    for text, gold_text in zip(predictions, golds):
        gold = parse_answer(gold_text, intents, slot_types)
        if gold is None:
            raise ValueError("Невалидная gold-разметка")
        pred = parse_answer(text, intents, slot_types)
        gold_slots = slot_counter(gold)
        pred_slots = Counter() if pred is None else slot_counter(pred)
        tp += sum((pred_slots & gold_slots).values())
        fp += sum((pred_slots - gold_slots).values())
        fn += sum((gold_slots - pred_slots).values())
        if pred is not None:
            valid += 1
            intent_ok = pred["intent"] == gold["intent"]
            correct_intent += int(intent_ok)
            exact += int(intent_ok and pred_slots == gold_slots)
    n = len(golds)
    denom = 2 * tp + fp + fn
    slot_f1 = 2 * tp / denom if denom else float(valid == n)
    intent_acc = correct_intent / n
    return {
        "quality": 0.5 * (intent_acc + slot_f1),
        "intent_accuracy": intent_acc,
        "slot_micro_f1": slot_f1,
        "frame_exact_match": exact / n,
        "valid_json_rate": valid / n,
        "n": n,
    }
