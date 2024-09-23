from pymilvus import MilvusClient, FieldSchema, CollectionSchema, DataType, Collection, utility, connections
from pymilvus import (
    AnnSearchRequest,
    WeightedRanker,
)
from milvus_model.hybrid import BGEM3EmbeddingFunction
import os

# Load the model 
ef = BGEM3EmbeddingFunction(use_fp16=False, device="cpu")
dense_dim = ef.dim["dense"]

#connect to milvus
col_name = "resumes_collection"

fields = [
    # Use auto generated id as primary key
    FieldSchema(
        name="id", dtype=DataType.INT64, is_primary=True, max_length=100
    ),
    FieldSchema(name="sparse", dtype=DataType.SPARSE_FLOAT_VECTOR),
    FieldSchema(name="dense", dtype=DataType.FLOAT_VECTOR, dim=dense_dim),
    # Store the original text to retrieve based on semantically distance
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=60000),
    FieldSchema(name="profession", dtype=DataType.VARCHAR, max_length=256),
    FieldSchema(name="file_name", dtype=DataType.VARCHAR, max_length=256),
    # Milvus now supports both sparse and dense vectors,
    # we can store each in a separate field to conduct hybrid search on both vectors
    
]
# Create a schema for the collection
schema = CollectionSchema(fields)

connections.connect(uri="./milvus.db")
col = Collection(col_name, schema, consistency_level="Strong")

# Enter your search query
query = input("Enter your search query: ")

# Generate embeddings for the query
query_embeddings = ef([query])
# print(query_embeddings)

# dense_search: only search across dense vector field
# sparse_search: only search across sparse vector field
# hybrid_search: search across both dense and vector fields with a weighted reranker
def dense_search(col, query_dense_embedding, limit=10):
    search_params = {"metric_type": "IP", "params": {}}
    res = col.search(
        [query_dense_embedding],
        anns_field="dense",
        limit=limit,
        output_fields=["text","file_name"],
        param=search_params,
    )[0]
    return [(hit.get("text"),hit.get("file_name")) for hit in res]


def sparse_search(col, query_sparse_embedding, limit=10):
    search_params = {
        "metric_type": "IP",
        "params": {},
    }
    res = col.search(
        [query_sparse_embedding],
        anns_field="sparse",
        limit=limit,
        output_fields=["text"],
        param=search_params,
    )[0]
    return [(hit.get("text"),hit.get("file_name")) for hit in res]


def hybrid_search(
    col,
    query_dense_embedding,
    query_sparse_embedding,
    sparse_weight=1.0,
    dense_weight=1.0,
    limit=10,
):
    dense_search_params = {"metric_type": "IP", "params": {}}
    dense_req = AnnSearchRequest(
        [query_dense_embedding], "dense", dense_search_params, limit=limit
    )
    sparse_search_params = {"metric_type": "IP", "params": {}}
    sparse_req = AnnSearchRequest(
        [query_sparse_embedding], "sparse", sparse_search_params, limit=limit
    )
    rerank = WeightedRanker(sparse_weight, dense_weight)
    res = col.hybrid_search(
        [sparse_req, dense_req], rerank=rerank, limit=limit, output_fields=["text","file_name"]
    )[0]
    return [(hit.get("text"),hit.get("file_name")) for hit in res]

dense_results = dense_search(col, query_embeddings["dense"][0], limit=3)
sparse_results = sparse_search(col, query_embeddings["sparse"][[0]], limit=3)
hybrid_results = hybrid_search(
    col,
    query_embeddings["dense"][0],
    query_embeddings["sparse"][[0]],
    sparse_weight=0.7,
    dense_weight=1.0,
    limit=3,
)

retrieved = [[res[0], res[1]] for res in hybrid_results]

context = "\n".join([line_with_distance[0] for line_with_distance in retrieved])

PROMPT = """
Use the following pieces of information enclosed in <context> tags to provide an answer to the question enclosed in <question> tags.
<context>
{context}
</context>
<question>
{question}
</question>
"""
from huggingface_hub import InferenceClient

repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"

llm_client = InferenceClient(model=repo_id, timeout=120)

prompt = PROMPT.format(context=retrieved, question=query)
answer = llm_client.text_generation(
    prompt,
    max_new_tokens=10000,
).strip()
print(answer)