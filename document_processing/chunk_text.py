import os
import json
import re

input_folder = "data/processed"
output_folder = "data/processed"

CHUNK_SIZE = 500
OVERLAP = 50

for filename in os.listdir(input_folder):

    if filename.endswith("_cleaned.txt"):

        input_path = os.path.join(input_folder, filename)

        with open(input_path, "r", encoding="utf-8") as file:
            text = file.read()

        # Get document name
        document_name = filename.replace("_cleaned.txt", ".pdf")

        # Split text using page markers
        pages = re.split(r"--- Page (\d+) ---", text)

        chunks = []

        for i in range(1, len(pages), 2):

            page_number = int(pages[i])
            page_text = pages[i + 1].strip()

            words = page_text.split()

            start = 0

            while start < len(words):

                chunk_words = words[start:start + CHUNK_SIZE]

                if chunk_words:
                    chunks.append({
                        "text": " ".join(chunk_words),
                        "metadata": {
                            "document": document_name,
                            "page": page_number
                        }
                    })

                start += CHUNK_SIZE - OVERLAP

        # Save chunks
        output_name = filename.replace("_cleaned.txt", "_chunks.json")
        output_path = os.path.join(output_folder, output_name)

        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(chunks, file, indent=2, ensure_ascii=False)

        print(f"Chunked: {filename} → {len(chunks)} chunks")

print("Chunking completed!")