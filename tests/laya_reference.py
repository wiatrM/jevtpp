#!/usr/bin/env python3
"""Independent Python reference for tokenizer/tensor parity checks."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer


def render(kind, criteria):
    if kind == "score":
        return [f"level {i}: {text}" for i, text in enumerate(criteria)]
    if kind == "noul":
        return ["false: no, the statement does not hold", "true: yes, the statement holds"]
    return criteria


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_directory", type=Path)
    args = parser.parse_args()
    root = args.model_directory
    agent = json.loads((root / "rl_agent_config.json").read_text())
    tokenizer_config = json.loads((root / "tokenizer/tokenizer_config.json").read_text())
    tokenizer = Tokenizer.from_file(str(root / "tokenizer/tokenizer.json"))
    special = {key: tokenizer.token_to_id(tokenizer_config[key])
               for key in ("cls_token", "sep_token", "mask_token", "pad_token")}
    mask_text = tokenizer_config["mask_token"]
    encode = lambda text: tokenizer.encode(text, add_special_tokens=False).ids
    state = '{"ticket":{"message":"Nie mogę się zalogować; cała firma jest zablokowana."},"customer":{"plan":"enterprise","affected_users":240},"service":{"sla_minutes":30,"workaround":false}}'
    questions = [
        ("choice", "integration.ticket.category", "Evaluate the customer support ticket using the supplied application context.\n\nWhich team should own this request?", ["Invoices, payments, or refund requests.", "Software bugs, crashes, or login failures.", "Upgrades, enterprise pricing, or new purchases.", "Unsolicited marketing or automated noise."]),
        ("noul", "integration.ticket.is_urgent", "Evaluate the customer support ticket using the supplied application context.\n\nDoes this require attention within one hour?", []),
        ("score", "integration.ticket.urgency_score", "Evaluate the customer support ticket using the supplied application context.\n\nRate the frustration level against the rubric.", ["Customer is patient and calm.", "Issue blocks work but has a workaround.", "Complete outage or severe frustration."]),
        ("noul", "integration.ticket.sentiment_probability", "Evaluate the customer support ticket using the supplied application context.\n\nThe customer is angry.", []),
    ]
    rows = []
    qtype_number = {"choice": 0, "score": 1, "noul": 2}
    for kind, _, question, criteria in questions:
        options = render(kind, criteria)
        scrub = lambda text: text.replace(mask_text, " ")
        head = encode(f"{kind} question: {scrub(question)}")
        option_ids = [[special["mask_token"], *encode(" " + scrub(option))[:48]] for option in options]
        option_budget = agent["head_max_len"] - sum(map(len, option_ids))
        if option_budget < 16:
            per = max(4, (agent["head_max_len"] - 16) // max(1, len(option_ids)))
            option_ids = [value[:per] for value in option_ids]
            option_budget = agent["head_max_len"] - sum(map(len, option_ids))
        sequence = [special["cls_token"], *head[:max(8, option_budget)], special["sep_token"]]
        markers = []
        for value in option_ids:
            markers.append(len(sequence)); sequence.extend(value)
        sequence.append(special["sep_token"])
        room = max(0, agent["max_len"] - len(sequence) - 1)
        sequence.extend(encode(scrub(state))[:room]); sequence.append(special["sep_token"])
        rows.append((sequence[:agent["max_len"]], markers, qtype_number[kind]))
    length = max(map(lambda row: len(row[0]), rows)); width = max(len(row[1]) for row in rows)
    ids = np.full((len(rows), length), special["pad_token"], dtype=np.int64)
    attention = np.zeros_like(ids); positions = np.zeros((len(rows), width), dtype=np.int64)
    marker_mask = np.zeros((len(rows), width), dtype=np.bool_); qtypes = np.zeros(len(rows), dtype=np.int64)
    for index, (sequence, markers, qtype) in enumerate(rows):
        ids[index, :len(sequence)] = sequence; attention[index, :len(sequence)] = 1
        positions[index, :len(markers)] = markers; marker_mask[index, :len(markers)] = True; qtypes[index] = qtype
    logits = ort.InferenceSession(str(root / "model.onnx"), providers=["CPUExecutionProvider"]).run(
        ["logits"], {"input_ids": ids, "attention_mask": attention, "marker_pos": positions,
                     "marker_mask": marker_mask, "qtype": qtypes})[0]
    names = ["choice", "noul", "score", "noul"]
    sizes = [4, 2, 3, 2]
    output = []
    for row, (name, size, qtype) in enumerate(zip(names, sizes, qtypes)):
        bucket = "2" if size <= 2 else "3-5" if size <= 5 else "6-10" if size <= 10 else "11+"
        temperature = agent.get("temperature_by_options", {}).get(f"{name}:{bucket}", agent["temperature"][qtype])
        values = logits[row, :size] / temperature
        probabilities = np.exp(values - values.max()); probabilities /= probabilities.sum()
        output.append(probabilities.tolist())
    print(json.dumps(output))


if __name__ == "__main__":
    main()
