import os
import json
import re

# === Paths ===
INPUT_FILE = "data/extracted/full_text.txt"
OUTPUT_FILE = "data/extracted/handbook_chunks.jsonl"

# === Config ===
CHUNK_SIZE = 450  # target word count per chunk
MIN_CHUNK_WORDS = 200

def clean_and_split_text(raw_text):
    """Cleans up raw PDF text and splits it into paragraph-based chunks."""
    paragraphs = raw_text.split("\n\n")
    chunks = []
    current_chunk = []

    for para in paragraphs:
        words = para.strip().split()
        if not words:
            continue

        current_chunk.extend(words)

        if len(current_chunk) >= CHUNK_SIZE:
            chunks.append(" ".join(current_chunk))
            current_chunk = []

    # Add final leftover
    if len(current_chunk) >= MIN_CHUNK_WORDS:
        chunks.append(" ".join(current_chunk))

    return chunks

def save_chunks_to_jsonl(chunks, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for i, chunk in enumerate(chunks):
            json.dump({"chunk_id": i+1, "text": chunk}, f)
            f.write("\n")

if __name__ == "__main__":
    print(f"📖 Reading from: {INPUT_FILE}")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        full_text = f.read()

    chunks = clean_and_split_text(full_text)
    print(f"✂️ Created {len(chunks)} chunks.")

    save_chunks_to_jsonl(chunks, OUTPUT_FILE)
    print(f"✅ Chunks saved to: {OUTPUT_FILE}")
