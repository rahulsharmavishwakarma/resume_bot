from pymilvus import MilvusClient, FieldSchema, CollectionSchema, DataType, Collection, utility, connections
from pymilvus import (
    AnnSearchRequest,
    WeightedRanker,
)
from milvus_model.hybrid import BGEM3EmbeddingFunction
from sentence_transformers import SentenceTransformer
import json
import os
from tqdm.autonotebook import tqdm, trange

# Load the model 
ef = BGEM3EmbeddingFunction(use_fp16=False, device="cuda")
dense_dim = ef.dim["dense"]

# Load resumes from the dataset.json file
with open("data.json", "r") as file:
    resumes = json.load(file)

# Prepare the data in the required format
data = []

for i, resume in tqdm(enumerate(resumes)):
    # Combine resume fields to create the full text
    professional_summary = resume.get("professional_summary", "")
    skills = resume.get("skills", "")
    work_history = resume.get("work_history", "")
    education = resume.get("education", "")
    text = professional_summary + " " + skills + " " + work_history + " " + education
    
    # Generate embedding for the full text
    embedding = ef([text])
    sparse_vector = embedding["sparse"][0]
    dense_vector = embedding["dense"][0]
    
    # Each entity contains id, vector, and metadata (without file_name)
    entity = {
        "id": i,
        "dense": dense_vector,  # Embedding for the resume text
        "sparse":sparse_vector,
        "text": text,  # Full text of the resume
        "profession": resume["profession"],  # Profession metadata
        "file_name":resume["file_name"]
    }
    
    data.append(entity)

# Now, 'data' contains the resumes with ids, vectors, and metadata (without file_name)
print(f"Data has {len(data)} entities, each with fields: {data[0].keys()}")
print(f"Vector dim: {len(data[0]['dense'])}")

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
# Connect to Milvus
client = MilvusClient("milvus.db")
# Create a collection
col_name = "resumes_collection"
if client.has_collection(collection_name=col_name):
    client.drop_collection(collection_name=col_name)
client.create_collection(
    collection_name=col_name,
    dimension=dense_dim,
    schema=schema,
)
#indexing
index_params = client.prepare_index_params()
index_params.add_index(
    field_name="sparse",
    index_name="sparse_inverted_index",
    index_type="SPARSE_INVERTED_INDEX",
    metric_type="IP"
)

index_params.add_index(
    field_name="dense",
    index_name="dense",
    index_type="AUTOINDEX",
    metric_type="IP"
)
client.create_index(collection_name=col_name, index_params=index_params)

res = client.insert(collection_name=col_name, data=data)

print(f"Total entities inserted: {res['insert_count']}")