import json
import os
import time
import tomllib
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pymongo import MongoClient, UpdateOne
from pymongo.operations import SearchIndexModel

load_dotenv()

INDEX_NAME = "vector_index"
SYSTEM_PROMPT_PATH = Path(__file__).parent / "config" / "system_prompt.toml"
GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"


def load_system_prompt(path=SYSTEM_PROMPT_PATH):
    with open(path, "rb") as prompt_file:
        config = tomllib.load(prompt_file)
    rules = "\n".join(f"- {item['rule']}" for item in config["guardrails"])
    return f"{config['instruction']}\n\nGuardrails:\n{rules}"


SYSTEM_PROMPT = load_system_prompt()


def load_golden_dataset(path=GOLDEN_DATASET_PATH):
    dataset = json.loads(path.read_text())
    for row in dataset:
        if (
            not row.get("question")
            or not row.get("reference_contexts")
            or not row.get("ground_truth")
        ):
            raise ValueError(
                f"Golden row is missing question, reference_contexts, or ground_truth: {row}"
            )
    return dataset


golden_dataset = load_golden_dataset()
documents = []
_seen_documents = set[Any]()
for row in golden_dataset:
    for text in row["reference_contexts"]:
        if text not in _seen_documents:
            _seen_documents.add(text)
            documents.append(text)

embedding_model = OpenAIEmbeddings(
    model=os.environ["LLM_MODEL"],
    api_key=os.environ["LLM_API_KEY"],
    base_url=os.environ["LLM_API_ENDPOINT"].removesuffix("/embeddings"),
    check_embedding_ctx_length=False,
    model_kwargs={"encoding_format": "float"},
)

chat_model = ChatOpenAI(
    model=os.environ["CHAT_MODEL"],
    api_key=os.environ["LLM_API_KEY"],
    base_url=os.environ["CHAT_API_ENDPOINT"].removesuffix("/chat/completions"),
)

mongo_client = MongoClient[Any](
    os.environ["MONGO_HOST"] + os.environ.get("MONGO_HOST_PARAMS", "")
)
collection = mongo_client[os.environ["MONGO_DB"]][os.environ["MONGO_COLLECTION"]]
collection.create_index("text", unique=True)


def store_documents(texts):
    # embedding = embedding_model.embed_query("hello worldxyxxyx")

    # print("embedding", len(embedding))
    vectors = embedding_model.embed_documents(texts)

    # collection.bulk_write([
    #     UpdateOne(
    #         {"text": text},
    #         {"$set": {"text": text, "embedding": vector, "model": embedding_model.model}},
    #         upsert=True,
    #     )
    #     for text, vector in zip(texts, vectors)
    # ])
    return len(vectors[0])


def ensure_vector_index(num_dimensions):
    for index in collection.list_search_indexes():
        if index["name"] == INDEX_NAME:
            return
    collection.create_search_index(
        SearchIndexModel(
            definition={
                "fields": [
                    {
                        "type": "vector",
                        "path": "embedding",
                        "numDimensions": num_dimensions,
                        "similarity": "cosine",
                    }
                ]
            },
            name=INDEX_NAME,
            type="vectorSearch",
        )
    )


def wait_until_queryable(timeout=120):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for index in collection.list_search_indexes():
            if index.get("name") == INDEX_NAME and index.get("queryable"):
                return
        time.sleep(2)
    raise TimeoutError(f"{INDEX_NAME} was not queryable within {timeout}s")


def similarity_search(query, k=2):
    query_vector = embedding_model.embed_query(query)
    results = collection.aggregate(
        [
            {
                "$vectorSearch": {
                    "index": INDEX_NAME,
                    "path": "embedding",
                    "queryVector": query_vector,
                    "numCandidates": max(20, k),
                    "limit": k,
                }
            },
            {
                "$project": {
                    "text": 1,
                    "score": {"$meta": "vectorSearchScore"},
                }
            },
        ]
    )
    return [(doc["text"], doc["score"]) for doc in results]


def evaluate_retrieval(cases, k=2):
    """Score retrieval with Hit@k, Recall@k, and mean reciprocal rank."""
    hit_count = 0
    recall_total = 0.0
    reciprocal_rank_total = 0.0
    rows = []

    for case in cases:
        matches = similarity_search(case["question"], k=k)
        retrieved = [text for text, _score in matches]
        relevant = set[Any](case["reference_contexts"])
        rank = next(
            (
                index + 1
                for index, text in enumerate[Any](retrieved)
                if text in relevant
            ),
            None,
        )
        found = sum(1 for text in retrieved if text in relevant)
        hit = rank is not None
        hit_count += int(hit)
        recall_total += found / len(relevant)
        reciprocal_rank_total += (1 / rank) if rank else 0.0
        rows.append(
            {
                "query": case["question"],
                "hit": hit,
                "rank": rank,
                "retrieved": retrieved,
            }
        )

    count = len(cases)
    return {
        "k": k,
        "hit_at_k": hit_count / count,
        "recall_at_k": recall_total / count,
        "mrr": reciprocal_rank_total / count,
        "rows": rows,
    }


def answer_question(query, matches):
    context = "\n".join(f"- {text}" for text, _score in matches)
    response = chat_model.invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"Context:\n{context}\n\nQuestion: {query}"),
        ]
    )
    return response.content


dimensions = store_documents(documents)
print("dimensions", dimensions)
ensure_vector_index(dimensions)
wait_until_queryable()

# query = "this claim saying the patient responsibility is 100, my current deductibel is 500 dollar, Do I need st..."

query = "my claim is about gene therapy, and wants to know why it is being rejected?"

matches = similarity_search(query, k=2)
print(answer_question(query, matches))
print()
for text, score in matches:
    print(f"{score:.4f} {text}")

# print()
# report = evaluate_retrieval(golden_dataset, k=2)
# print(
#     f"retrieval k={report['k']}  "
#     f"hit@{report['k']}={report['hit_at_k']:.2f}  "
#     f"recall@{report['k']}={report['recall_at_k']:.2f}  "
#     f"mrr={report['mrr']:.2f}"
# )
# for row in report["rows"]:
#     mark = "hit" if row["hit"] else "miss"
#     rank = row["rank"] if row["rank"] else "-"
#     print(f"  {mark} rank={rank} {row['query']}")
#     for text in row["retrieved"]:
#         print(f"    {text}")
