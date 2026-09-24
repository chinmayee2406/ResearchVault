# ResearchVault

### Multi-Paper Research Intelligence RAG System

ResearchVault is a local Retrieval-Augmented Generation (RAG) research assistant designed to analyze, retrieve, compare, and synthesize information across multiple research papers.

It combines semantic retrieval, BM25 lexical retrieval, Reciprocal Rank Fusion (RRF), cross-encoder reranking, and a locally hosted LLM to generate grounded answers with page-level supporting evidence.

---

## Overview

Reading and comparing multiple research papers manually can make it difficult to:

- locate relevant evidence quickly
- compare methodologies across papers
- identify differences in evaluation strategies
- synthesize findings across multiple sources
- trace generated answers back to the original papers

ResearchVault addresses this by creating a temporary research workspace where users can upload multiple PDFs and ask questions about them.

The system retrieves relevant evidence from the uploaded papers before generating an answer, reducing dependence on the language model's parametric knowledge.

---

## Key Features

### Multi-Paper Question Answering

Upload multiple research papers and ask questions across the entire research session.

Example:

> What are the main challenges addressed by these papers?

---

### Hybrid Retrieval

ResearchVault combines two complementary retrieval approaches:

**Semantic Retrieval**

Uses Sentence Transformers to identify conceptually relevant passages.

**BM25 Retrieval**

Uses lexical matching to retrieve passages containing important query terms.

The two rankings are combined using **Reciprocal Rank Fusion (RRF)**.

