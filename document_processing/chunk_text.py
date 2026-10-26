import os
import json

input_folder = "data/processed"
output_folder = "data/processed"

CHUNK_SIZE = 500
OVERLAP = 50

os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if not filename.endswith("_cleaned.txt"):
        continue

    input_path = os.path.join(input_folder, filename)

    with open(input_path, "r", encoding="utf-8") as file:
        text = file.read().strip()

    words = text.split()
    chunks = []

    start = 0
    while start < len(words):
        chunk_words = words[start:start + CHUNK_SIZE]

        chunks.append({
            "text": " ".join(chunk_words),
            "metadata": {
                "document": filename.removesuffix("_cleaned.txt")
            }
        })

        start += CHUNK_SIZE - OVERLAP

    output_name = filename.replace(
        "_cleaned.txt", "_chunks.json"
    )
    output_path = os.path.join(output_folder, output_name)

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(chunks, file, indent=2, ensure_ascii=False)

    print(f"Chunked: {filename} -> {len(chunks)} chunks")

print("Chunking completed!")