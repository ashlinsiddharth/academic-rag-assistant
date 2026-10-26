"""Interactive smoke test for retrieval over data/processed JSON chunks."""

from __future__ import annotations

import argparse

from retrieval import DEFAULT_DATA_DIR, FAISSRetriever, load_chunks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question", help="Question to search; prompts interactively if omitted.")
    parser.add_argument("--top-k", type=int, default=3, help="Number of chunks to retrieve (default: 3).")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), help="Directory containing processed chunk JSON files.")
    parser.add_argument("--model", default="all-MiniLM-L6-v2", help="Sentence Transformers model name.")
    args = parser.parse_args()

    question = args.question or input("Enter an academic question: ").strip()
    chunks = load_chunks(args.data_dir)
    retriever = FAISSRetriever(model_name=args.model)
    retriever.build_index(chunks)
    results = retriever.search(question, top_k=args.top_k)

    print(f"\nQuestion: {question}\n")
    for rank, result in enumerate(results, start=1):
        print(f"{rank}. {result['text']}")
        source = result.get("source", "Unknown source")
        details = [f"Source: {source}"]
        if result.get("page") is not None:
            details.append(f"Page: {result['page']}")
        if result.get("chunk") is not None:
            details.append(f"Chunk: {result['chunk']}")
        if result.get("section") is not None:
            details.append(f"Section: {result['section']}")
        print("   " + " | ".join(details))
        print(f"   Similarity score: {result['score']:.4f}")


if __name__ == "__main__":
    main()
