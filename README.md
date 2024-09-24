# Resume Retrieval and Question-Answering System

## Project Description

The Resume Retrieval and Question-Answering System is an advanced application designed to streamline the hiring process through efficient resume retrieval and intelligent question answering. The system leverages the following key technologies:


1. **Vector Database: Milvus**:
   - Used for storing and retrieving resume embeddings
   - Supports both sparse and dense vector representations
   - Enables hybrid search capabilities

2. **Embedding Generation: BGEM3EmbeddingFunction**:
   - Generates both sparse and dense embeddings for resume content
   - Supports CPU and GPU acceleration

3. **Large Language Model: Mistral AI's Mixtral-8x7B-Instruct-v0.1**:
   - Utilized for generating human-like responses to queries based on retrieved resume contexts

4. **Natural Language Processing Libraries**:
   - PyMilvus for interacting with the Milvus database
   - Hugging Face's InferenceClient for interfacing with the large language model

5. **Development Environment**:
   - Python-based implementation
   - Supports both CPU and GPU processing

This tech stack combines the power of vector databases for efficient semantic search with advanced language models for intelligent query processing, creating a robust system for resume analysis and candidate matching.

### Key Components

1. **Data Preprocessing**:
   - The project begins with a preprocessing script (`preprocess.py`) that extracts and structures resume data from PDF files. This includes identifying key sections such as professional summaries, skills, work history, and education.
   - The extracted text is tokenized and normalized, ensuring consistency and accuracy for subsequent analysis.
   - Dense and sparse embeddings are generated using machine learning models, facilitating effective semantic search capabilities.

2. **Vector Database Integration**:
   - The `vector_db.py` module is responsible for setting up and managing a Milvus vector database, which is optimized for storing and retrieving high-dimensional data like embeddings.
   - It creates collections, defines schemas for the resume data (including fields for dense and sparse vectors), and inserts processed resume data into the database.
   - This architecture allows for hybrid search capabilities, combining both dense and sparse vectors to enhance retrieval accuracy.
   - Employs BGEM3EmbeddingFunction for generating embeddings.
   - Utilizes Milvus for efficient vector storage and retrieval.

3. **Search and Retrieval Pipeline**:
   - The `pipeline.py` file serves as the core of the retrieval mechanism. It includes methods for performing dense, sparse, and hybrid searches against the Milvus database.
   - The system intelligently ranks the results based on semantic similarity, ensuring that users receive the most relevant resumes in response to their queries.
   - It also includes functionality to generate embeddings for user queries, allowing for natural language searches that yield meaningful results.
   - Utilizes a large language model (Mixtral-8x7B) for generating answers based on retrieved contexts.
   - Uses both sparse and dense vector search capabilities of Milvus.

4. **User Interface**:
   - The application is presented through a web interface built with Gradio (`app.py`), offering a chat-like experience for users to input queries and view results.
   - Users can ask about specific skills, job titles, or other qualifications, and the system retrieves relevant resumes while providing concise, context-aware answers using a language model.
   - The interface displays retrieved documents and allows for follow-up questions, enabling an interactive exploration of candidate profiles.

### Overall Functionality

The Resume Retrieval and Question-Answering System enhances the recruitment process by:
- **Streamlining Candidate Search**: Quickly identifies and retrieves relevant resumes based on specific criteria.
- **Improving Candidate Insights**: Provides detailed answers about candidates’ qualifications and experiences, aiding decision-making for hiring managers.
- **Facilitating Interactive Exploration**: Allows users to ask follow-up questions, ensuring a comprehensive understanding of the candidate pool.

### Use Cases

- HR professionals and recruiters can leverage the system to efficiently sift through large volumes of resumes, saving time and improving the quality of candidate selection.
- Organizations can use the system to better match candidates to job requirements based on detailed insights derived from resume content.

### Step 1: Clone the Repository

First, clone the repository containing the project files to your local machine:

```bash
git clone https://github.com/rahulsharmavishwakarma/resume_bot
cd resume_bot
```

### Step 2: Set Up the Python Environment

1. **Create a Virtual Environment** (optional but recommended):

   ```bash
   python -m venv venv
   ```

2. **Activate the Virtual Environment**:
   - On Windows:

     ```bash
     venv\Scripts\activate
     ```
   - On macOS/Linux:

     ```bash
     source venv/bin/activate
     ```

### Step 3: Install Required Packages

Install the required packages listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

### Step 4: Prepare the Resume Data

1. **Organize Your PDF Resumes**:
   - Place your PDF resumes into a directory structure, categorized by profession. For example:

     ```
     /data/
         /software_engineer/
             resume1.pdf
             resume2.pdf
         /data_scientist/
             resume1.pdf
             resume2.pdf
     ```

2. **Run the Preprocessing Script**:
   - Execute the `preprocess.py` script to extract and structure resume data:

     ```bash
     python preprocess.py
     ```
   - This will create a `data.json` file containing the structured resume data.

### Step 5: Populate the Vector Database

1. **Run the Vector Database Script**:
   - Execute the `vector_db.py` script to set up the Milvus vector database and insert the processed resume data:

     ```bash
     python vector_db.py
     ```
   - Ensure that the Milvus service is running before executing this step.

### Step 6: Set Up the Retrieval Pipeline

1. **Run the Pipeline Script** (optional):
   - If needed, test the pipeline functionality using `pipeline.py` to ensure the search mechanisms work correctly:

     ```bash
     python pipeline.py
     ```

### Step 7: Launch the Web Application

1. **Run the Gradio App**:
   - Finally, execute the `app.py` script to start the web interface:

     ```bash
     python app.py
     ```

2. **Access the Application**:
   - Open a web browser and navigate to the provided Gradio URL (usually `http://127.0.0.1:7860`) to interact with the application.

## Using the Application

- Enter your search queries related to candidates' skills, experiences, or qualifications.
- Review the retrieved documents and answers provided by the system.
- Ask follow-up questions to explore the candidate profiles further.