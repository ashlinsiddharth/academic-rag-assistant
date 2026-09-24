import fitz
import os

input_folder = "data/raw"
output_folder = "data/processed"

os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):

    if filename.endswith(".pdf"):

        pdf_path = os.path.join(input_folder, filename)
        doc = fitz.open(pdf_path)

        text = ""

        for page_number, page in enumerate(doc):
            text += f"\n--- Page {page_number + 1} ---\n"
            text += page.get_text()

        output_name = filename.replace(".pdf", ".txt")
        output_path = os.path.join(output_folder, output_name)

        with open(output_path, "w", encoding="utf-8") as file:
            file.write(text)

        print(f"Done: {filename}")

print("All PDFs processed!")