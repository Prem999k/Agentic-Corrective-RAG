https://arxiv.org/pdf/2401.15884

# Corrective RAG (CRAG) with Groq GPT-OSS 20B

A **Corrective Retrieval-Augmented Generation (CRAG)** system built with **LangGraph** that evaluates retrieved document chunks, detects weak retrieval, and uses **Tavily web search** as a fallback when the local knowledge is insufficient.

## Tech Stack

- **LLM:** Groq GPT-OSS 20B
- **Embeddings:** BAAI/bge-base-en-v1.5
- **Vector Store:** FAISS
- **Workflow:** LangGraph
- **Web Search:** Tavily
- **Framework:** LangChain
- **Language:** Python

---

## Architecture

```text
                         USER QUESTION
                               |
                               v
                      +----------------+
                      |    RETRIEVE    |
                      |     FAISS      |
                      +-------+--------+
                              |
                              v
                    +--------------------+
                    | DOCUMENT EVALUATOR |
                    |   GPT-OSS 20B      |
                    +---------+----------+
                              |
                 +------------+------------+
                 |            |            |
                 v            v            v
             CORRECT      AMBIGUOUS    INCORRECT
                 |            |            |
                 |            +------+\    |
                 |                   | \   |
                 v                   |  v  v
              REFINE                 | QUERY REWRITE
                 |                   | GPT-OSS 20B
                 |                   |      |
                 |                   |      v
                 |                   |   TAVILY
                 |                   |      |
                 +-------------------+------+
                              |
                              v
                           REFINE
                              |
                              v
                          GENERATE
                       GPT-OSS 20B
                              |
                              v
                            ANSWER
```

---

## How CRAG Works

### 1. Document Ingestion

The application automatically reads files from the `documents/` folder.

Supported file types:

```text
PDF
DOCX
TXT
MD
CSV
JSON
HTML
HTM
```

Supported link files:

```text
.url
.webloc
.links
.urls
```

Example:

```text
documents/
├── machine_learning.pdf
├── deep_learning.docx
├── notes.txt
├── research.md
└── links.txt
```

A `links.txt` file can contain:

```text
https://example.com/article1
https://example.com/article2
```

---

### 2. Chunking

Documents are split into smaller chunks using:

```text
chunk_size = 900
chunk_overlap = 150
```

This improves retrieval by creating smaller searchable pieces of information.

---

### 3. Embeddings

The project uses:

```text
BAAI/bge-base-en-v1.5
```

The embedding model runs locally and converts document chunks into numerical vectors.

---

### 4. FAISS Retrieval

The generated embeddings are stored in **FAISS**.

For each question, the system retrieves the top 4 similar chunks:

```text
k = 4
```

---

### 5. Document Evaluation

Each retrieved chunk is evaluated using:

```text
Groq GPT-OSS 20B
```

The model gives a relevance score between:

```text
0.0 → 1.0
```

Thresholds used:

```text
UPPER_TH = 0.7
LOWER_TH = 0.3
```

The retrieval is classified into three states.

#### CORRECT

At least one retrieved document chunk scores above `0.7`.

```text
CORRECT
   ↓
REFINE
   ↓
GENERATE
```

#### INCORRECT

All retrieved chunks score below `0.3`.

```text
INCORRECT
   ↓
QUERY REWRITE
   ↓
TAVILY SEARCH
   ↓
REFINE
   ↓
GENERATE
```

#### AMBIGUOUS

The retrieved information is neither clearly correct nor clearly incorrect.

```text
AMBIGUOUS
   ↓
LOCAL DOCUMENTS + WEB RESULTS
   ↓
REFINE
   ↓
GENERATE
```

---

### 6. Query Rewriting

When local retrieval is insufficient, GPT-OSS 20B rewrites the question into a concise web-search query.

Example:

```text
User Question:
What are the latest developments in generative AI?

Rewritten Query:
latest generative AI developments
```

The rewritten query is sent to Tavily.

---

### 7. Web Search

The project uses:

```text
Tavily
```

to retrieve additional information from the web when required by the CRAG workflow.

---

### 8. Context Refinement

The retrieved context is broken into individual sentences.

GPT-OSS 20B determines which sentences are directly relevant to the user's question.

Only relevant sentences are retained for final generation.

---

### 9. Final Answer

GPT-OSS 20B generates the answer using only the refined context.

If the available information is insufficient, the system returns:

```text
I don't know.
```

---

## Project Structure

```text
Agentic_CRAG/
│
├── documents/
│   ├── file1.pdf
│   ├── file2.docx
│   ├── notes.txt
│   └── links.txt
│
├── CRAG.py
├── CRAG.ipynb
├── requirements.txt
├── .gitignore
├── .env
└── README.md
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Prem999k/Agentic-Corrective-RAG.git
cd Agentic-Corrective-RAG
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Or with `uv`:

```bash
uv pip install -r requirements.txt --link-mode=copy
```

---

## Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
TAVILY_API_KEY=your_tavily_api_key
```

Do **not** commit `.env` to GitHub.

---

## Running the Project

Run:

```bash
python CRAG.py
```

You will get:

```text
======================================================================
CRAG - Groq GPT-OSS 20B
Type your question.
Type 'exit' to stop.
======================================================================

Question:
```

Enter a question:

```text
Question: What is overfitting?
```

The system then performs:

```text
Question
   ↓
FAISS Retrieval
   ↓
Document Evaluation
   ↓
CRAG Routing
   ↓
Refinement / Web Search
   ↓
Final Answer
```

You can continue asking multiple questions without restarting the application.

Type:

```text
exit
```

to stop.

---

## Example Questions

Questions related to local documents:

```text
What is batch normalization?
```

```text
Explain gradient descent.
```

```text
What is overfitting?
```

Questions that may trigger web fallback:

```text
What are the latest developments in generative AI?
```

```text
What is the latest version of Python?
```

---

## Key Features

- Corrective Retrieval-Augmented Generation
- Retrieval quality evaluation
- Automatic CRAG routing
- Local document search
- Web-search fallback
- Query rewriting
- Sentence-level context filtering
- BGE local embeddings
- FAISS vector similarity search
- LangGraph workflow orchestration
- GPT-OSS 20B through Groq
- Multiple document formats
- URL and web-page ingestion
- Interactive question answering

---

## Why CRAG?

Traditional RAG can generate an answer even when the retrieved information is poor.

CRAG adds a retrieval evaluation step:

```text
Retrieve
   ↓
Evaluate
   ↓
Is retrieval reliable?
   |
   +---- YES ----> Refine → Generate
   |
   +---- NO -----> Rewrite → Web Search
                         ↓
                       Refine
                         ↓
                      Generate
```

This makes the system more robust against poor or incomplete retrieval.

---

## Technologies

| Component | Technology |
|---|---|
| LLM | Groq GPT-OSS 20B |
| Embeddings | BAAI/bge-base-en-v1.5 |
| Vector Store | FAISS |
| Workflow | LangGraph |
| Framework | LangChain |
| Web Search | Tavily |
| PDF Processing | PyPDFLoader / pypdf |
| DOCX Processing | python-docx |
| Web Parsing | BeautifulSoup |
| Language | Python |

---

## Future Improvements

- Persistent FAISS index
- Reranking model
- Metadata filtering
- Source citations
- Conversational memory
- PostgreSQL chat history
- Streamlit chat interface
- Streaming responses
- Multi-user document collections
- RAG evaluation metrics

---

## Author

**Prem Kumar**

GitHub:  
https://github.com/Prem999k

Project Repository:  
https://github.com/Prem999k/Agentic-Corrective-RAG
