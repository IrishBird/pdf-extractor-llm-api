# Extracts structured data and vector embeddings from PDFs into MongoDB, orchestrated with LangGraph and served via a REST API.

End-to-end pipeline: extract one PDF document with an LLM into a predefined schema, persist structured sections and embeddings, then answer questions over that data through a LangGraph-orchestrated REST API.

** This repo only include partial scripts **.

# Wiki
https://github.com/IrishBird/pdf-extractor-llm-api/wiki

---

## How It Works

1. **PDF → Schema (LLM):** The loader reads pages, splits the PDF into overlapping chunks, and calls the LLM concurrently. Each chunk is filled into the predefined benefit schema (`src/model/source_schema.py`). Results are merged, duplicate sections are collapsed, and the document is validated to make sure all pages have been processed by the LLM. Failures go to the RDS error table and are not loaded.
2. **Schema → Mongo + Vector DB:** Valid sections are upserted into Mongo keyed by `businessKey + sectionName` (parsed from the PDF filename). The same document is split into passages, embedded, and written to the vector Mongo collection for similarity search.
3. **REST API → LangGraph → Tools → Mongo:** `POST /v1/assistant` runs a LangGraph (`pre_llm → assistant → optional tools` loop). Bound tools (e.g., `get_benefit_data`) embed the user query and run `$vectorSearch` against Mongo, then the assistant answers from tool results.

```text
PDF ─▶ LLM extract ─▶ benefit schema ─▶ upsert ─▶ Mongo (sections)
                                      └─▶ embed ─▶ Vector Mongo
POST /v1/assistant ─▶ LangGraph ─▶ tools ───┘
```

### Example: LangGraph Orchestrates the Mongo Vector Tool

```python
# api/graphs/chat_graph.py - assistant - tools loop
def build_graph(self) -> StateGraph[ChatState]:
    graph = StateGraph(state_schema=ChatState)
    graph.add_node("pre_llm", self.pre_llm)
    graph.add_node("assistant", self.assistant)  # LLM with tools bound
    graph.add_node("tools", self.execute_tool)   # ToolNode - get_benefit_data
    graph.add_edge(START, "pre_llm")
    graph.add_conditional_edges(
        "pre_llm", self.route_pre_llm, {"assistant": "assistant", END: END}
    )
    graph.add_conditional_edges(
        "assistant", self.should_continue, {"tools": "tools", END: END}
    )
    graph.add_edge("tools", "assistant")  # tool results back to the model
    return graph
```

The tool embeds the question and searches vector Mongo:

```python
# api/components/tools.py - get_benefit_data - $vectorSearch
results = self.collection.aggregate([
    {
        "$vectorSearch": {
            "index": self.index_name,
            "path": "embedding",
            "queryVector": embedding_model.embed_query(words),
            "numCandidates": num_candidates,
            "limit": self.limits,
        }
    },
    {"$project": {"text": 1, "score": {"$meta": "vectorSearchScore"}}},
])
```

---

## Running the Extractor

### Local One-Shot (Explicit File)

```bash
source .venv/bin/activate
uv pip install pip
python -m pip --version

# Run against a local PDF
python src/pdf-extractor_main.py --pdf File1.pdf
```

### SQS Worker (S3 Event-Driven)

`S3 Object Created → SQS → worker` long-polls the queue, reads the bucket/key from the event, downloads the PDF, runs the same pipeline, and deletes the message on success (failed loads leave the message for retry).

```bash
# Requires SQS_QUEUE_URL (and AWS credentials / IAM role)
python src/pdf-extractor_main.py --once  # One poll cycle, then exit
```

> **Note:** Running the container entrypoint with no `--pdf` flag starts this SQS loop continuously.

| Variable | Purpose |
| :--- | :--- |
| `SQS_QUEUE_URL` | Queue that receives S3 notifications (required in SQS mode) |
| `AWS_REGION` | Region for SQS/S3 clients (default: `us-east-1`) |
| `SQS_WAIT_SECONDS` | Long-poll wait in seconds, max 20 (default: `20`) |
| `SQS_VISIBILITY_TIMEOUT` | Visibility timeout while processing in seconds (default: `900`) |
| `SQS_MAX_MESSAGES` | Messages per receive call (default: `1`) |
| `PDF_WORK_DIR` | Local download directory (default: system temp) |

---

## Evaluate Vector Mongo (`src/eval`)

After the loader writes embeddings to `VECTOR_MONGO_*`, score inventory quality and retrieval:

```bash
# Structural checks (row counts, required fields, dims, sections)
python src/eval/eval_vector_main.py --mode inventory

# Retrieval Hit@k / Recall@k / MRR against src/eval/golden_dataset.json
python src/eval/eval_vector_main.py --mode retrieval --k 5

# Run all checks with filters and export report
python src/eval/eval_vector_main.py --mode all --account 111117 --product 702301 --output documents/output/eval_report.json
```

Edit `src/eval/golden_dataset.json` to add questions with `expected_sectionNames`, `expected_path_contains`, and/or `reference_text_contains`.

---

## API (`api/app.py`)

### Local Smoke Test

```bash
source .venv/bin/activate
python api/app.py
```

Uvicorn starts with reload enabled and listens on `http://0.0.0.0:8000`. Startup logs look like:

```text
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete.
```

> **Note:** There is no root route—`GET /` (and probes like `GET /json/version`) return `404` as expected. Use the assistant endpoint: **`POST /v1/assistant`**.

Alternatively, start with Uvicorn directly:

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```

*Stop the server with `Ctrl+C` (`Ctrl+Z` only suspends the process).*

---

## Environment Variables (`.env`)

`.env` holds every variable the loader reads and is listed in `.gitignore`.

### MongoDB
* `MONGO_HOST`
* `MONGO_HOST_PARAMS`
* `MONGO_DB`
* `MONGO_COLLECTION` *(optional; defaults to the value in config)*

### RDS
Set either `DB_SECRET_NAME` **or** all of the following:
* `DB_HOST`
* `DB_PORT`
* `DB_DATABASE`
* `DB_USERNAME`
* `DB_PASSWORD`

### LLM
Required. Extraction calls the model specified by `LLM_MODEL`:
* `LLM_API_ENDPOINT`
* `LLM_MODEL`
* `LLM_API_KEY`

Optional fallback:
* `LLM_FALLBACK_ENDPOINT`
* `LLM_FALLBACK_MODEL`

---

## Configuration & Environment Overrides

All core runtime settings live in `config/pdf-extractor_config.py` and can be overridden using `LLM_`-prefixed environment variables:

* **PDF Chunking & Concurrency**
  * `page_overlap` (`LLM_PAGE_OVERLAP`)
  * `concurrency` (`LLM_CONCURRENCY`)
  * `chunk_pause_seconds` (`LLM_CHUNK_PAUSE_SECONDS`)
  * `pdf_per_chunk` (`LLM_PDF_PER_CHUNK`)
* **Model & Generation Limits**
  * `max_tokens` (`LLM_MAX_TOKENS`)
  * `max_rounds` (`LLM_MAX_ROUNDS`)
  * `Context Window`
  * `TPM (Tokens Per Minute)`
  * `RPM (Requests Per Minute)`
* **HTTP, Retries & Resilience**
  * `max_http_attempts` (`LLM_MAX_HTTP_ATTEMPTS`)
  * HTTP timeout, `429` rate-limit waits, server/network backoff, and circuit-breaker thresholds
* **Checkpointing & Storage Behavior**
  * During execution, progress is written to `<pdf-name>.partial.json` alongside the source PDF. Once the MongoDB upsert succeeds, this temporary file is automatically deleted.


# Embedding and Retrieval Evaluation

Create a golden dataset and evaluation framework to benchmark embedding models and retrieval performance within the RAG pipeline. The goal is to ensure that relevant documents and chunks are consistently retrieved for user queries and that retrieval quality does not regress over time.

---

## Golden Dataset

The golden dataset contains representative user queries and their expected retrieval results.

### Dataset Format

```json
    {
      "question": "What do I pay in-network for Allergy Serum (dispensed by Primary Care Physician in the office)?",
      "accountNumber": "00001",
      "productNumber": "PROD001",
      "expected_sectionNames": [
        "medicalBenefits"
      ],
      "expected_path_contains": [
        "medicalBenefits.medical.tiers[0].categories[0]"
      ],
      "reference_text_contains": [
        "Allergy Serum (dispensed by Primary Care Physician in the office)"
      ]
    }
```


### Dataset Coverage

The golden dataset should include:

- Exact keyword searches
- Semantic similarity queries
- Synonym and acronym variations
- Multi-document retrieval scenarios
- Multi-hop retrieval scenarios
- Long conversational queries
- Ambiguous queries
- Negative test cases with no expected matches

### Recommended Distribution

| Query Type | Percentage |
|------------|------------|
| Exact Match | 40% |
| Semantic Search | 30% |
| Multi-Hop Retrieval | 15% |
| Synonyms / Acronyms | 10% |
| Negative Cases | 5% |

---

## Embedding Evaluation

The embedding evaluation measures whether semantically similar content is mapped close together in vector space.


### Embedding Metrics

- Cosine Similarity
- Recall@K
- Precision@K
- Mean Reciprocal Rank (MRR)
- NDCG

---

## Retrieval Evaluation

For each query in the golden dataset:

1. Generate embeddings.
2. Execute retrieval against the vector store.
3. Compare retrieved results against expected results.
4. Compute evaluation metrics.
5. Track results over time.


#### Result
## Retrieval Evaluation Results

| Metric | Score | Description |
|----------|--------|-------------|
| Recall@5 | 72.5% | 72.5% of relevant documents were retrieved within the top 5 results. |
| MRR | 0.622 | The first relevant result appeared on average around rank 1.6. |
| Hit@5 | 0.73 | A relevant document was found in the top 5 results for 73% of queries. |

### Summary

The retrieval system achieved:

- **Recall@5:** 72.5%
- **MRR:** 0.622
- **Hit@5:** 0.73

These results indicate that the system retrieves relevant content for most queries, although there is room for improvement in both retrieval coverage and ranking quality

---

## Evaluation Metrics

### Recall@K

Measures whether relevant documents are retrieved.

```text
Recall@K =
Retrieved Relevant Documents
----------------------------------
Total Relevant Documents
```

Target:

```text
Recall@5 > 90%
```

---

### Precision@K

Measures how many retrieved documents are relevant.

```text
Precision@K =
Relevant Retrieved Documents
----------------------------------
K
```

Target:

```text
Precision@5 > 70%
```

---

### Mean Reciprocal Rank (MRR)

Measures how high the first relevant result appears.

```text
MRR = 1 / Rank
```

Examples:

```text
Rank 1 → 1.0
Rank 2 → 0.5
Rank 5 → 0.2
```

Target:

```text
MRR > 0.80
```

---

### NDCG@K

Measures ranking quality when multiple relevant results exist.

Target:

```text
NDCG@10 > 0.85
```

---

## Benchmark Comparisons

Evaluate and compare:

### Embedding Models

- text-embedding-3-small
- text-embedding-3-large

### Chunk
