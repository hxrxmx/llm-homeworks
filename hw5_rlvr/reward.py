import json
from collections import Counter
from functools import lru_cache


def canonical(value):
    if isinstance(value, dict):
        return {k: canonical(v) for k, v in sorted(value.items())}
    if isinstance(value, list):
        return [canonical(v) for v in value]
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return value.strip() if isinstance(value, str) else value


def key(value):
    return json.dumps(canonical(value), sort_keys=True, ensure_ascii=False)


def f1(pred, gold):
    pred, gold = Counter(pred), Counter(gold)
    if not pred and not gold:
        return 1.0
    return 2 * sum((pred & gold).values()) / max(1, sum(pred.values()) + sum(gold.values()))


@lru_cache(maxsize=1)
def tokenizer():
    from transformers import AutoTokenizer
    from prepare import MODEL, MODEL_REVISION
    return AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REVISION)


@lru_cache(maxsize=1)
def parser():
    from vllm.tool_parsers.qwen3coder_tool_parser import Qwen3CoderToolParser
    return Qwen3CoderToolParser(tokenizer())


@lru_cache(maxsize=128)
def request(tools_json):
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    data = {"model": "Qwen/Qwen3.5-2B", "messages": [{"role": "user", "content": "score"}]}
    tools = json.loads(tools_json)
    if tools:
        data["tools"] = tools
    return ChatCompletionRequest.model_validate(data)


def parse_calls(text, tools):
    text = text.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()
    tool_parser = parser()
    req = request(json.dumps(tools, sort_keys=True))
    tool_parser.tools = req.tools or []
    try:
        parsed = tool_parser.extract_tool_calls(text, req)
        calls = [{"name": c.function.name, "arguments": json.loads(c.function.arguments)}
                 for c in parsed.tool_calls]
    except (ValueError, TypeError, KeyError):
        return [], False
    valid = bool(text)
    for opening, closing in [("<tool_call>", "</tool_call>"), ("<function=", "</function>"),
                             ("<parameter=", "</parameter>")]:
        valid = valid and text.count(opening) == text.count(closing)
    valid = valid and text.count("<tool_call>") == len(calls)
    if not calls and any(tag in text for tag in ["<tool", "<function", "<parameter"]):
        valid = False
    return calls, valid


def components(text, tools, reference):
    from jsonschema import Draft7Validator

    calls, valid = parse_calls(text, tools)
    exact = valid and Counter(key(c) for c in calls) == Counter(key(c) for c in reference)
    schemas = {t["function"]["name"]: t["function"]["parameters"] for t in tools}
    schema_ok = valid
    for call in calls:
        name = call["name"]
        schema_ok = schema_ok and name in schemas
        if name in schemas:
            schema_ok = schema_ok and Draft7Validator(schemas[name]).is_valid(call["arguments"])
    names = f1([c["name"] for c in calls], [c["name"] for c in reference])
    pred_args = [(c["name"], k, key(v)) for c in calls for k, v in c["arguments"].items()]
    gold_args = [(c["name"], k, key(v)) for c in reference for k, v in c["arguments"].items()]
    args = f1(pred_args, gold_args) if pred_args or gold_args else names
    if not reference:
        shaped = float(valid and not calls)
    elif not valid or not calls:
        shaped = 0.0
    else:
        shaped = .1 * schema_ok + .2 * names + .2 * args + .5 * exact
    return {"exact": float(exact), "shaped": float(shaped), "valid": bool(valid),
            "schema": bool(schema_ok), "name_f1": names, "argument_f1": args, "calls": calls}


def make_reward(mode):
    eos_ids = {tokenizer().eos_token_id, tokenizer().pad_token_id}
    def reward(completions, tools_json, reference_json, completion_ids=None, **kwargs):
        scores = []
        for i, (text, tools, reference) in enumerate(zip(completions, tools_json, reference_json)):
            if completion_ids is not None and (not completion_ids[i] or completion_ids[i][-1] not in eos_ids):
                scores.append(0.0)
                continue
            score = components(text, json.loads(tools), json.loads(reference))
            scores.append(score[mode])
        return scores
    reward.__name__ = mode
    return reward


def check():
    from prepare import read_calls
    assert read_calls('[Convert(from="USD", to="EUR", amount=5000)]', ["Convert"])[0]["arguments"]["from"] == "USD"
    tools = [{"type": "function", "function": {"name": "weather", "parameters": {
        "type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]
    }}}]
    gold = [{"name": "weather", "arguments": {"city": "Paris"}}]
    good = '<tool_call>\n<function=weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>'
    cases = [good, good.replace("Paris", "London"), good.replace("</tool_call>", ""),
             "Which city do you mean?", good + good]
    result = [components(text, tools, gold) for text in cases]
    assert result[0]["exact"] == result[0]["shaped"] == 1
    assert 0 < result[1]["shaped"] < 1 and result[1]["exact"] == 0
    assert result[2]["shaped"] == 0 and result[3]["shaped"] == 0
    assert result[4]["exact"] == 0 and result[4]["shaped"] < 1
    assert components(cases[3], tools, [])["shaped"] == 1
    assert components(good, tools, [])["shaped"] == 0
    assert components("No suitable tool.", [], [])["exact"] == 1
    empty_tools = [{"type": "function", "function": {"name": name, "parameters": {"type": "object", "properties": {}}}}
                   for name in ["a", "b"]]
    wrong = '<tool_call>\n<function=b>\n</function>\n</tool_call>'
    assert components(wrong, empty_tools, [{"name": "a", "arguments": {}}])["argument_f1"] == 0
    assert make_reward("exact")(["Which city?"], [json.dumps(tools)], ["[]"], completion_ids=[[1, 2]]) == [0.0]
    assert Counter([key({"a": 1}), key({"b": 2})]) == Counter([key({"b": 2.0}), key({"a": 1.0})])
    print("reward checks passed")


if __name__ == "__main__":
    check()
