from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

model_id = "NousResearch/Nous-Hermes-2-Mistral-7B-DPO"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto"
)

import json
from tqdm import tqdm


# Input and output file paths
input_path = "handbook_chunks.jsonl"   # <- uploaded file
output_path = "qa_output.jsonl"

# Read input chunks
with open(input_path, "r", encoding="utf-8") as f:
    chunks = [json.loads(line.strip()) for line in f]

# Generate QA pairs
with open(output_path, "w", encoding="utf-8") as out_f:
    for chunk in tqdm(chunks):
        if "text" not in chunk:
            continue

        context = chunk["text"]
        print(f"Processing context: {context[:100]}...")  # Print the first 100 characters of context for debugging

        prompt = f"""[INST] You are a helpful assistant. Based on the following context, generate 3 question-answer pairs that help someone understand the content.

Context:
\"\"\"{context}\"\"\" [/INST]"""

        # Tokenize the input prompt and ensure it fits within the model's max length
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)

        # Generate model output
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            top_p=0.9,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Generated text: {generated_text[:200]}...")  # Print the first 200 characters of the generated output

        # Remove the prompt from the output (we only need the generated QA pairs)
        generated_text = generated_text.split("Context:")[1].strip()  # Split to remove the prompt part

        # Basic Q-A extraction
        qa_pairs = []
        lines = generated_text.split("\n")
        current_q = ""
        current_a = ""

        for line in lines:
            if line.strip().lower().startswith("question"):
                if current_q and current_a:
                    qa_pairs.append({"question": current_q, "answer": current_a})
                current_q = line.strip()[9:].strip()  # Remove "Question: " part
                current_a = ""
            elif line.strip().lower().startswith("answer") and current_q:
                current_a = line.strip()[7:].strip()  # Remove "Answer: " part

        if current_q and current_a:
            qa_pairs.append({"question": current_q, "answer": current_a})

        if qa_pairs:
            json.dump({"context": context, "qa_pairs": qa_pairs}, out_f, ensure_ascii=False)
            out_f.write("\n")

print("QA generation completed.")