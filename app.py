import gradio as gr
from pymilvus import FieldSchema, DataType, Collection, connections, CollectionSchema
from pymilvus import AnnSearchRequest, WeightedRanker
from milvus_model.hybrid import BGEM3EmbeddingFunction
from huggingface_hub import InferenceClient

# Connect to Milvus
connections.connect(uri="./milvus.db")

# Load the model
ef = BGEM3EmbeddingFunction(use_fp16=False, device="cpu")
dense_dim = ef.dim["dense"]

# Collection name
col_name = "resumes_collection"

# Define collection schema
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
    FieldSchema(name="sparse", dtype=DataType.SPARSE_FLOAT_VECTOR),
    FieldSchema(name="dense", dtype=DataType.FLOAT_VECTOR, dim=dense_dim),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=60000),
    FieldSchema(name="profession", dtype=DataType.VARCHAR, max_length=256),
    FieldSchema(name="file_name", dtype=DataType.VARCHAR, max_length=256)
]

# Create a schema for the collection
schema = CollectionSchema(fields)
col = Collection(col_name, schema, consistency_level="Strong")

# Hugging Face LLM Client
repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
llm_client = InferenceClient(model=repo_id, timeout=120)

# Optimized prompt template
PROMPT = """
You are an expert at analyzing resumes and extracting meaningful insights. Below is a set of relevant resume information from a database. Use this information to answer the following query in a precise and informative manner.

Resume Data:
<context>
{context}
</context>

Query:
<question>
{question}
</question>

Answer the query based on the provided resume data. Your response should be clear and directly address the question using any relevant information.
"""

# Search functions
def dense_search(col, query_dense_embedding, limit=10):
    search_params = {"metric_type": "IP", "params": {}}
    res = col.search(
        [query_dense_embedding],
        anns_field="dense",
        limit=limit,
        output_fields=["text", "file_name"],
        param=search_params,
    )[0]
    return [(hit.get("text"), hit.get("file_name")) for hit in res]

def sparse_search(col, query_sparse_embedding, limit=10):
    search_params = {"metric_type": "IP", "params": {}}
    res = col.search(
        [query_sparse_embedding],
        anns_field="sparse",
        limit=limit,
        output_fields=["text", "file_name"],
        param=search_params,
    )[0]
    return [(hit.get("text"), hit.get("file_name")) for hit in res]

def hybrid_search(col, query_dense_embedding, query_sparse_embedding, sparse_weight=1.0, dense_weight=1.0, limit=10):
    dense_search_params = {"metric_type": "IP", "params": {}}
    dense_req = AnnSearchRequest([query_dense_embedding], "dense", dense_search_params, limit=limit)
    
    sparse_search_params = {"metric_type": "IP", "params": {}}
    sparse_req = AnnSearchRequest([query_sparse_embedding], "sparse", sparse_search_params, limit=limit)
    
    rerank = WeightedRanker(sparse_weight, dense_weight)
    res = col.hybrid_search([sparse_req, dense_req], rerank=rerank, limit=limit, output_fields=["text", "file_name"])[0]
    return [(hit.get("text"), hit.get("file_name")) for hit in res]

# Function for Gradio interface
def search_resumes(query, search_type, sparse_weight=0.7, dense_weight=1.0, limit=3):
    # Generate embeddings for the query
    query_embeddings = ef([query])
    
    # Perform search based on the selected method
    if search_type == "Hybrid Search":
        results = hybrid_search(col, query_embeddings["dense"][0], query_embeddings["sparse"][0], sparse_weight, dense_weight, limit)
    elif search_type == "Dense Search":
        results = dense_search(col, query_embeddings["dense"][0], limit)
    else:
        results = sparse_search(col, query_embeddings["sparse"][0], limit)
    
    # Format retrieved results for display
    retrieved = [[res[0], res[1]] for res in results]
    context = "\n".join([line_with_distance[0] for line_with_distance in retrieved])

    # Generate a response from the LLM based on the context
    prompt = PROMPT.format(context=context, question=query)
    answer = llm_client.text_generation(prompt, max_new_tokens=10000).strip()
    return answer, context, retrieved

# Follow-up question handler
def follow_up(query, context):
    # Reuse the context from the initial search and answer the follow-up question
    prompt = PROMPT.format(context=context, question=query)
    follow_up_answer = llm_client.text_generation(prompt, max_new_tokens=10000).strip()
    return follow_up_answer

# Gradio chat-style interface
with gr.Blocks() as app:
    gr.Markdown("### Resume Bot - Resume Retrieval and Question-Answering")
    
    chatbot = gr.Chatbot(label="Chat with Resume Expert", height=500)

    with gr.Row():
        query_input = gr.Textbox(label="Your question", placeholder="Ask a question related to resumes...", lines=1)
        search_type = gr.Radio(["Hybrid Search", "Dense Search", "Sparse Search"], label="Search Method", value="Hybrid Search")

    sparse_weight = gr.Slider(label="Sparse Weight", minimum=0, maximum=1, step=0.1, value=0.7)
    dense_weight = gr.Slider(label="Dense Weight", minimum=0, maximum=1, step=0.1, value=1.0)
    limit = gr.Slider(label="Limit", minimum=1, maximum=10, step=1, value=3)

    submit_button = gr.Button("Ask")
    
    # DataFrame to display retrieved documents
    retrieved_docs = gr.DataFrame(label="Retrieved Resume Snippets and Filenames", headers=["Resume Snippet", "Filename"])

    # Initialize context state
    state = gr.State()

    # Handle initial query and search
    def initial_search(chat_history, query, search_type, sparse_weight, dense_weight, limit):
        # Generate the answer, search context, and retrieve documents
        answer, context, retrieved = search_resumes(query, search_type, sparse_weight, dense_weight, limit)
        # Append the user's query and LLM's answer to the chat history
        chat_history.append(("User", query))
        chat_history.append(("AI", answer))
        return chat_history, context, retrieved

    # Handle follow-up question
    def handle_follow_up(chat_history, follow_up_query, context):
        follow_up_answer = follow_up(follow_up_query, context)
        chat_history.append(("User", follow_up_query))
        chat_history.append(("AI", follow_up_answer))
        return chat_history

    # Bind the search function to the submit button
    submit_button.click(fn=initial_search, 
                        inputs=[chatbot, query_input, search_type, sparse_weight, dense_weight, limit], 
                        outputs=[chatbot, state, retrieved_docs])

    # Bind follow-up functionality to the text input submit
    query_input.submit(fn=handle_follow_up, 
                       inputs=[chatbot, query_input, state], 
                       outputs=[chatbot])

# Launch the app
app.launch(share=True)
