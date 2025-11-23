import json

with open("train.jsonl", "r", encoding="utf-8") as infile:
    lines = [json.loads(line) for line in infile]

with open("data_array.json", "w", encoding="utf-8") as outfile:
    json.dump(lines, outfile, ensure_ascii=False, indent=2)
