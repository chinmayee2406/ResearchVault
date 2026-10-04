# ResearchVault — Multi-Paper Research Intelligence RAG System
ResearchVault is a multi-paper Retrieval-Augmented Generation (RAG) system for querying and comparing research papers.
It processes PDFs, extracts and chunks their content, and creates semantic embeddings for retrieval.
It combines semantic search and BM25 using Reciprocal Rank Fusion (RRF) to improve evidence retrieval.
A cross-encoder reranker selects the most relevant evidence before passing it to a local LLM.
The system generates evidence-grounded answers with page-level citations and supports multi-paper comparison.
## Problem Statement
Research papers contain large volumes of information spread across many pages, making manual analysis and comparison time-consuming. Traditional keyword search can miss semantically relevant information, while relying only on an LLM can produce answers that are not grounded in the source documents. A system is needed that can efficiently retrieve relevant evidence from multiple papers and generate answers that can be traced back to their sources.
## Solution
ResearchVault builds an end-to-end RAG pipeline that processes and indexes research papers, retrieves relevant evidence using semantic search and BM25, combines the results using RRF, and improves ranking using a cross-encoder. The selected evidence is passed to a local LLM to generate grounded answers with page-level source references. A paper-aware comparison pipeline also allows information from multiple research papers to be analyzed side-by-side.
## Tech Stack
- Language: Python
- RAG: Retrieval-Augmented Generation
- Embeddings: Sentence Transformers — all-MiniLM-L6-v2
- Vector Database: ChromaDB
- Lexical Retrieval: BM25
- Hybrid Retrieval: Reciprocal Rank Fusion (RRF)
- Reranking: Cross-Encoder — ms-marco-MiniLM-L-6-v2
- LLM: Ollama — llama3.2:3b
- PDF Processing: PyMuPDF
- Backend: FastAPI
- Frontend: Streamlit
- Evaluation: Recall@5, MRR, answer relevance, faithfulness, citation accuracy
## Architecture
<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/2c246c3a-77b1-4e62-a3a3-b71c656677e9" />
