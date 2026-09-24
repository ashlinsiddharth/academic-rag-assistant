import os
import re

input_folder = "data/processed"
output_folder = "data/processed"

for filename in os.listdir(input_folder):

    if filename.endswith(".txt") and not filename.endswith("_cleaned.txt"):

        input_path = os.path.join(input_folder, filename)

        with open(input_path, "r", encoding="utf-8") as file:
            text = file.read()

        # Remove extra spaces
        text = re.sub(r"[ \t]+", " ", text)

        # Remove too many blank lines
        text = re.sub(r"\n\s*\n+", "\n\n", text)

        # Remove spaces at the beginning/end of lines
        text = "\n".join(line.strip() for line in text.splitlines())

        # Save cleaned text
        output_name = filename.replace(".txt", "_cleaned.txt")
        output_path = os.path.join(output_folder, output_name)

        with open(output_path, "w", encoding="utf-8") as file:
            file.write(text)

        print(f"Cleaned: {filename}")

print("Cleaning completed!")