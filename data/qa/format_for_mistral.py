import json

input_path = "data/qa/qa_dataset_mistral.jsonl"
output_path = "data/qa/mistral_finetune.jsonl"

with open(input_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
    for line in fin:
        item = json.loads(line)
        context = item["context"]
        for qa in item["qa_pairs"]:
            instruction = qa["question"].strip()
            output = qa["answer"].strip()
            fout.write(json.dumps({
                "instruction": instruction,
                "input": context,
                "output": output
            }, ensure_ascii=False) + "\n")

print(f"✅ Formatted dataset saved to {output_path}")