# academic-rag-assistant
RAG-Based Intelligent Academic Information Assistant for University Students

## Retrieval component

`retrieval.py` accepts Monish's processed chunks as strings or dictionaries such
as `{"text": "Students can borrow up to five books.", "source": "rules.pdf", "page": 3}`.
It returns the relevant chunks with their source details and a similarity score.

```python
from retrieval import FAISSRetriever

retriever = FAISSRetriever()
retriever.build_index(processed_chunks)
retriever.save("data/faiss_index")

relevant_chunks = retriever.search("How many books can I borrow?", top_k=3)
```
