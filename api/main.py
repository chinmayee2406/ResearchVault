from pathlib import Path
from typing import Annotated

import re
import shutil
import uuid

import chromadb
import ollama
import pymupdf

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

# ============================================================
# CONFIGURATION
# ============================================================

TEMP_STORAGE_DIR = Path("data/temp_sessions")

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LLM_MODEL_NAME = "lfm2.5:8b"
COMPARISON_CANDIDATES_PER_PAPER = 12
COMPARISON_EVIDENCE_PER_PAPER = 4

TEMP_STORAGE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# MODELS
# ============================================================

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL_NAME
)

cross_encoder = CrossEncoder(
    RERANKER_MODEL_NAME
)


# ============================================================
# CHROMADB
# ============================================================

chroma_client = chromadb.PersistentClient(
    path="data/chroma"
)


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="ResearchVault API",
    version="3.1.0",
    description="Temporary multi-paper AI research workspace"
)


# ============================================================
# SCHEMAS
# ============================================================

class AskRequest(BaseModel):
    session_id: str
    question: str


class EvidenceSource(BaseModel):
    paper_name: str
    page_number: int
    passage: str


class AnswerResponse(BaseModel):
    answer: str
    sources: list[EvidenceSource]


# ============================================================
# TOKENIZATION
# ============================================================

def tokenize(text: str):
    """
    Tokenize text for BM25 retrieval.
    """

    text = text.lower()

    text = re.sub(
        r"[^\w\s]",
        "",
        text
    )

    return text.split()


# ============================================================
# PAPER NAME HELPERS
# ============================================================

def clean_paper_name(filename: str) -> str:
    """
    Convert an uploaded filename into a clean paper name.
    """

    name = Path(filename).stem.strip()

    if not name:
        name = "Research Paper"

    return name


def make_unique_paper_name(
    base_name: str,
    used_names: set[str]
) -> str:
    """
    Prevent chunk ID collisions when two uploaded PDFs
    have the same filename.
    """

    if base_name not in used_names:
        return base_name

    counter = 2

    while f"{base_name}_{counter}" in used_names:
        counter += 1

    return f"{base_name}_{counter}"


# ============================================================
# PDF CHUNKING
# ============================================================

def extract_chunks_from_pdf(
    pdf_path: Path,
    paper_name: str
):
    """
    Extract text from a PDF and split it into overlapping chunks.

    Each chunk receives a stable paper-specific ID:

        PaperName_chunk_0
        PaperName_chunk_1
        PaperName_chunk_2
        ...
    """

    document = pymupdf.open(
        pdf_path
    )

    all_chunks = []

    paper_chunk_id = 0

    try:

        for page_index in range(
            len(document)
        ):

            page = document[
                page_index
            ]

            text = page.get_text(
                "text"
            ).strip()

            if not text:
                continue

            words = text.split()

            start = 0

            while start < len(words):

                end = min(
                    start + CHUNK_SIZE,
                    len(words)
                )

                chunk_words = words[
                    start:end
                ]

                chunk_text = " ".join(
                    chunk_words
                ).strip()

                if chunk_text:

                    all_chunks.append(
                        {
                            "paper": paper_name,

                            "page": (
                                page_index + 1
                            ),

                            "chunk_id": (
                                f"{paper_name}"
                                f"_chunk_"
                                f"{paper_chunk_id}"
                            ),

                            "text": chunk_text
                        }
                    )

                    paper_chunk_id += 1

                if end >= len(words):
                    break

                start = (
                    end - CHUNK_OVERLAP
                )

    finally:

        document.close()

    return all_chunks


# ============================================================
# SESSION HELPERS
# ============================================================

def get_collection_name(
    session_id: str
):
    return f"session_{session_id}"


def get_session_collection(
    session_id: str
):

    collection_name = (
        get_collection_name(
            session_id
        )
    )

    try:

        return chroma_client.get_collection(
            name=collection_name
        )

    except Exception:

        raise HTTPException(
            status_code=404,
            detail=(
                "This research session does not "
                "exist or has expired."
            )
        )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "service": "ResearchVault API"
    }


# ============================================================
# MULTI-PAPER UPLOAD
# ============================================================

@app.post("/session/upload")
async def upload_session_papers(
    files: Annotated[
        list[UploadFile],
        File(
            description="Upload one or more research paper PDF files"
        )
    ]
):

    # --------------------------------------------------------
    # Validate upload
    # --------------------------------------------------------

    if not files:

        raise HTTPException(
            status_code=400,
            detail="Please upload at least one PDF."
        )

    # --------------------------------------------------------
    # Validate every file
    # --------------------------------------------------------

    for file in files:

        if not file.filename:

            raise HTTPException(
                status_code=400,
                detail="One of the uploaded files has no filename."
            )

        if not file.filename.lower().endswith(".pdf"):

            raise HTTPException(
                status_code=400,
                detail=(
                    f"'{file.filename}' is not a PDF. "
                    "Only PDF files are supported."
                )
            )

    # --------------------------------------------------------
    # Create ONE temporary research session
    # --------------------------------------------------------

    session_id = str(
        uuid.uuid4()
    )

    session_dir = (
        TEMP_STORAGE_DIR /
        session_id
    )

    session_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Track uploaded papers
    # --------------------------------------------------------

    used_paper_names = set()

    uploaded_papers = []

    all_chunks = []

    # --------------------------------------------------------
    # Save + process EVERY uploaded PDF
    # --------------------------------------------------------

    try:

        for file_index, file in enumerate(files):

            base_name = clean_paper_name(
                file.filename
            )

            paper_name = make_unique_paper_name(
                base_name,
                used_paper_names
            )

            used_paper_names.add(
                paper_name
            )

            # Preserve PDF extension
            pdf_filename = (
                f"{paper_name}.pdf"
            )

            pdf_path = (
                session_dir /
                pdf_filename
            )

            # ------------------------------------------------
            # Save PDF temporarily
            # ------------------------------------------------

            with open(
                pdf_path,
                "wb"
            ) as buffer:

                shutil.copyfileobj(
                    file.file,
                    buffer
                )

            # ------------------------------------------------
            # Extract chunks
            # ------------------------------------------------

            chunks = extract_chunks_from_pdf(
                pdf_path,
                paper_name
            )

            if not chunks:

                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"No readable text could be extracted "
                        f"from '{file.filename}'."
                    )
                )

            # ------------------------------------------------
            # Add chunks to shared session
            # ------------------------------------------------

            all_chunks.extend(
                chunks
            )

            uploaded_papers.append(
                {
                    "paper_name": paper_name,
                    "original_filename": file.filename,
                    "chunk_count": len(chunks)
                }
            )

    except HTTPException:

        shutil.rmtree(
            session_dir,
            ignore_errors=True
        )

        raise

    except Exception as exc:

        shutil.rmtree(
            session_dir,
            ignore_errors=True
        )

        raise HTTPException(
            status_code=500,
            detail=(
                f"Could not process uploaded papers: {exc}"
            )
        )

    # --------------------------------------------------------
    # Make sure we actually have content
    # --------------------------------------------------------

    if not all_chunks:

        shutil.rmtree(
            session_dir,
            ignore_errors=True
        )

        raise HTTPException(
            status_code=400,
            detail=(
                "No readable research content was found "
                "in the uploaded PDFs."
            )
        )

    # ========================================================
    # CREATE ONE TEMPORARY CHROMA COLLECTION
    # ========================================================

    collection_name = (
        get_collection_name(
            session_id
        )
    )

    try:

        collection = (
            chroma_client
            .get_or_create_collection(
                name=collection_name
            )
        )

        # ----------------------------------------------------
        # Documents
        # ----------------------------------------------------

        documents = [
            chunk["text"]
            for chunk in all_chunks
        ]

        # ----------------------------------------------------
        # Stable IDs
        # ----------------------------------------------------

        ids = [
            chunk["chunk_id"]
            for chunk in all_chunks
        ]

        # ----------------------------------------------------
        # Metadata
        # ----------------------------------------------------

        metadatas = [

            {
                "paper": chunk["paper"],
                "page": chunk["page"],
                "chunk_id": chunk["chunk_id"]
            }

            for chunk in all_chunks
        ]

        # ----------------------------------------------------
        # Embeddings
        # ----------------------------------------------------

        embeddings = embedding_model.encode(
            documents,
            normalize_embeddings=True
        )

        # ----------------------------------------------------
        # Add everything to ONE session collection
        # ----------------------------------------------------

        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings.tolist()
        )

    except Exception as exc:

        try:

            chroma_client.delete_collection(
                name=collection_name
            )

        except Exception:
            pass

        shutil.rmtree(
            session_dir,
            ignore_errors=True
        )

        raise HTTPException(
            status_code=500,
            detail=(
                f"Could not create research session: {exc}"
            )
        )

    # ========================================================
    # RETURN SESSION INFORMATION
    # ========================================================

    paper_names = [
        paper["paper_name"]
        for paper in uploaded_papers
    ]

    return {

        "session_id": session_id,

        "paper_names": paper_names,

        "paper_count": len(
            paper_names
        ),

        "papers": uploaded_papers,

        "chunk_count": len(
            all_chunks
        ),

        "message": (
            "Research papers uploaded successfully."
        )
    }


# ============================================================
# COMPARISON HELPERS
# ============================================================

def normalize_paper_name(name: str) -> str:
    """Normalize a paper name for robust question matching."""

    return re.sub(
        r"[^a-z0-9]+",
        "",
        name.lower()
    )


def paper_name_aliases(paper_name: str) -> set[str]:
    """
    Build safe aliases from the uploaded paper filename.

    This intentionally uses the filename/stem supplied by the user
    rather than inventing paper names from retrieved content.
    """

    stem = Path(paper_name).stem.strip().lower()

    aliases = {
        normalize_paper_name(stem)
    }

    # Also support filenames such as:
    #   T-bench
    #   T bench
    #   t_bench
    tokenized = re.findall(r"[a-z0-9]+", stem)

    if tokenized:
        aliases.add("".join(tokenized))

    # A single distinctive filename token is useful for names such
    # as AgentDojo, T-bench, YouStruQ, etc. Avoid generic words.
    generic_tokens = {
        "paper", "research", "study", "survey", "benchmark",
        "method", "approach", "analysis", "final", "draft",
        "version", "new", "updated"
    }

    for token in tokenized:
        if len(token) >= 4 and token not in generic_tokens:
            aliases.add(token)

    return {
        alias
        for alias in aliases
        if len(alias) >= 3
    }


def paper_is_explicitly_named(
    question: str,
    paper_name: str
) -> bool:
    """Return True only when the question explicitly names a paper."""

    normalized_question = normalize_paper_name(question)

    for alias in paper_name_aliases(paper_name):
        if alias in normalized_question:
            return True

    # Handle filenames containing multiple words when the compact
    # form is not sufficient.
    question_tokens = set(
        re.findall(r"[a-z0-9]+", question.lower())
    )
    paper_tokens = set(
        re.findall(
            r"[a-z0-9]+",
            Path(paper_name).stem.lower()
        )
    )

    generic_tokens = {
        "paper", "research", "study", "survey", "benchmark",
        "method", "approach", "analysis", "final", "draft",
        "version", "new", "updated"
    }

    distinctive_tokens = {
        token
        for token in paper_tokens
        if len(token) >= 4
        and token not in generic_tokens
    }

    return bool(
        distinctive_tokens
        and distinctive_tokens.intersection(question_tokens)
    )


def detect_comparison_papers(
    question: str,
    session_papers: list[str]
):
    """
    Detect comparison intent and resolve ONLY the papers explicitly
    named by the user.

    For explicit comparisons such as:
        "What is the difference between T-bench and AgentDojo?"
    the returned list contains only those two uploaded papers.

    For implicit comparisons such as:
        "Compare these papers"
    all papers in the current temporary session are used.
    """

    question_lower = question.lower()

    comparison_terms = [
        "compare",
        "comparison",
        "difference",
        "differences",
        "differ",
        "versus",
        "vs",
        "between",
        "each paper",
        "these papers",
        "both papers",
        "how do they",
        "how are they",
        "different approaches",
        "different methods"
    ]

    if not any(
        term in question_lower
        for term in comparison_terms
    ):
        return None

    explicit_matches = [
        paper_name
        for paper_name in session_papers
        if paper_is_explicitly_named(
            question,
            paper_name
        )
    ]

    # CRITICAL: never replace explicitly requested papers with
    # arbitrary papers from the session.
    if len(explicit_matches) >= 2:
        return explicit_matches

    implicit_terms = [
        "these papers",
        "both papers",
        "each paper",
        "the papers",
        "compare the papers",
        "compare these"
    ]

    if any(
        term in question_lower
        for term in implicit_terms
    ):
        return session_papers

    # The user explicitly named papers, but ResearchVault could
    # resolve fewer than two of them. Returning None prevents the
    # system from silently comparing the wrong uploaded papers.
    return None


def build_paper_indices(
    metadatas: list[dict]
):
    """Build a paper -> document-index mapping."""

    paper_indices = {}

    for index, metadata in enumerate(
        metadatas
    ):

        paper_name = metadata[
            "paper"
        ]

        paper_indices.setdefault(
            paper_name,
            []
        ).append(index)

    return paper_indices


def retrieve_comparison_results(
    question: str,
    query_embedding,
    documents: list[str],
    metadatas: list[dict],
    collection,
    bm25_scores,
    comparison_papers: list[str]
):
    """
    Retrieve comparison evidence independently for every requested paper.

    The comparison pipeline deliberately combines:
    1. early-paper anchor chunks (title/abstract/introduction),
    2. dimension-focused semantic retrieval,
    3. paper-local BM25,
    4. RRF fusion,
    5. cross-encoder reranking.

    This is important because a generic query such as
    "difference between X and Y" often retrieves only the most distinctive
    paper or only an attack example. Comparison needs evidence about the
    papers themselves, not just the most similar passage.
    """

    paper_indices = build_paper_indices(metadatas)
    comparison_results = []

    # These queries deliberately target different parts of a research paper.
    focus_queries = [
        f"{question} paper objective purpose research goal benchmark",
        f"{question} problem formulation methodology method approach framework tasks",
        f"{question} evaluation experiments results metrics performance",
        f"{question} contribution findings limitations significance",
    ]

    # Encode all focus queries once. This avoids repeatedly invoking the
    # embedding model for every paper/query combination.
    focus_embeddings = embedding_model.encode(
        focus_queries,
        normalize_embeddings=True
    )

    for paper_name in comparison_papers:

        indices = paper_indices.get(paper_name, [])

        if not indices:
            continue

        # --------------------------------------------------------
        # 1. ANCHOR EVIDENCE
        # --------------------------------------------------------
        # The first chunks usually contain the title, abstract and/or
        # introduction. They are valuable for identifying what the paper
        # itself is about, especially for benchmark papers.
        anchor_indices = indices[:2]

        anchor_results = []

        for index in anchor_indices:
            anchor_results.append(
                {
                    "score": 0.0,
                    "metadata": metadatas[index],
                    "document": documents[index],
                    "source_type": "paper_anchor",
                }
            )

        # --------------------------------------------------------
        # 2. SEMANTIC RETRIEVAL FOR THIS PAPER ONLY
        # --------------------------------------------------------

        semantic_rrf = {}

        for focus_embedding in focus_embeddings:

            semantic_results = collection.query(
                query_embeddings=[focus_embedding.tolist()],
                n_results=min(
                    COMPARISON_CANDIDATES_PER_PAPER,
                    len(indices)
                ),
                where={"paper": paper_name}
            )

            semantic_documents = semantic_results["documents"][0]
            semantic_metadatas = semantic_results["metadatas"][0]

            for rank, metadata in enumerate(semantic_metadatas):

                chunk_id = metadata["chunk_id"]

                if chunk_id not in semantic_rrf:
                    semantic_rrf[chunk_id] = {
                        "score": 0.0,
                        "metadata": metadata,
                        "document": semantic_documents[rank],
                        "source_type": "semantic",
                    }

                semantic_rrf[chunk_id]["score"] += (
                    1 / (60 + rank + 1)
                )

        # --------------------------------------------------------
        # 3. PAPER-LOCAL BM25
        # --------------------------------------------------------

        comparison_bm25_query = (
            f"{paper_name} {question} "
            "objective purpose problem methodology method approach "
            "framework benchmark tasks evaluation experiments results "
            "metrics performance contribution findings limitations"
        )

        comparison_bm25_tokens = tokenize(
            comparison_bm25_query
        )

        paper_bm25 = BM25Okapi(
            [
                tokenize(documents[i])
                for i in indices
            ]
        )

        paper_bm25_raw_scores = paper_bm25.get_scores(
            comparison_bm25_tokens
        )

        paper_bm25_scores = {
            index: float(paper_bm25_raw_scores[position])
            for position, index in enumerate(indices)
        }

        ranked_bm25_indices = sorted(
            indices,
            key=lambda i: paper_bm25_scores[i],
            reverse=True
        )[:COMPARISON_CANDIDATES_PER_PAPER]

        # --------------------------------------------------------
        # 4. RRF FUSION
        # --------------------------------------------------------

        fused_results = {}

        for chunk_id, result in semantic_rrf.items():
            fused_results[chunk_id] = {
                "score": result["score"],
                "metadata": result["metadata"],
                "document": result["document"],
                "source_type": result["source_type"],
            }

        for rank, index in enumerate(ranked_bm25_indices):

            metadata = metadatas[index]
            chunk_id = metadata["chunk_id"]

            if chunk_id not in fused_results:
                fused_results[chunk_id] = {
                    "score": 0.0,
                    "metadata": metadata,
                    "document": documents[index],
                    "source_type": "bm25",
                }

            fused_results[chunk_id]["score"] += (
                1 / (60 + rank + 1)
            )

        fused_results = sorted(
            fused_results.values(),
            key=lambda x: x["score"],
            reverse=True
        )[:COMPARISON_CANDIDATES_PER_PAPER]

        # --------------------------------------------------------
        # 5. CROSS-ENCODER RERANKING
        # --------------------------------------------------------

        if fused_results:

            rerank_query = (
                f"Paper: {paper_name}. "
                f"Comparison question: {question}. "
                "Find evidence that explains this paper's own "
                "objective, problem, method, evaluation, results, "
                "contribution, or limitations."
            )

            rerank_pairs = [
                (
                    rerank_query,
                    result["document"]
                )
                for result in fused_results
            ]

            rerank_scores = cross_encoder.predict(
                rerank_pairs
            )

            for result, score in zip(
                fused_results,
                rerank_scores
            ):
                result["rerank_score"] = float(score)

            reranked_results = sorted(
                fused_results,
                key=lambda x: x["rerank_score"],
                reverse=True
            )
        else:
            reranked_results = []

        # --------------------------------------------------------
        # 6. BUILD DIVERSE, PAPER-OWNED EVIDENCE
        # --------------------------------------------------------
        # Keep early-paper anchors plus the strongest distinct pages.
        # This prevents a concrete attack example from crowding out
        # the paper's own abstract/method/evaluation evidence.
        selected = []
        seen_chunk_ids = set()
        seen_pages = set()

        for result in anchor_results:
            chunk_id = result["metadata"]["chunk_id"]
            page = result["metadata"]["page"]

            if chunk_id not in seen_chunk_ids:
                selected.append(result)
                seen_chunk_ids.add(chunk_id)
                seen_pages.add(page)

        for result in reranked_results:
            chunk_id = result["metadata"]["chunk_id"]
            page = result["metadata"]["page"]

            if chunk_id in seen_chunk_ids:
                continue

            # Prefer a new page so the comparison receives different
            # sections of the paper.
            if page not in seen_pages:
                selected.append(result)
                seen_chunk_ids.add(chunk_id)
                seen_pages.add(page)

            if len(selected) >= COMPARISON_EVIDENCE_PER_PAPER:
                break

        # If there are not enough distinct pages, fill from reranked results.
        if len(selected) < COMPARISON_EVIDENCE_PER_PAPER:
            for result in reranked_results:
                chunk_id = result["metadata"]["chunk_id"]

                if chunk_id in seen_chunk_ids:
                    continue

                selected.append(result)
                seen_chunk_ids.add(chunk_id)

                if len(selected) >= COMPARISON_EVIDENCE_PER_PAPER:
                    break

        comparison_results.extend(selected)

    # ------------------------------------------------------------
    # HARD PROVENANCE CHECK
    # ------------------------------------------------------------
    # An explicit comparison is allowed to continue ONLY when every
    # requested paper actually contributed evidence.
    retrieved_papers = {
        result["metadata"]["paper"]
        for result in comparison_results
    }

    expected_papers = set(comparison_papers)

    if retrieved_papers != expected_papers:

        missing_papers = [
            paper
            for paper in comparison_papers
            if paper not in retrieved_papers
        ]

        raise HTTPException(
            status_code=422,
            detail=(
                "Comparison stopped to protect source accuracy. "
                f"Requested papers: {comparison_papers}. "
                f"Evidence found for: {sorted(retrieved_papers) or ['none']}. "
                f"Missing evidence for: {missing_papers}. "
                "ResearchVault will not substitute another uploaded paper."
            )
        )

    return comparison_results


# ============================================================
# ASK QUESTION
# ============================================================

@app.post(
    "/session/ask",
    response_model=AnswerResponse
)
def ask_question(
    request: AskRequest
):

    # --------------------------------------------------------
    # Get temporary collection
    # --------------------------------------------------------

    collection = get_session_collection(
        request.session_id
    )

    # --------------------------------------------------------
    # Get stored chunks
    # --------------------------------------------------------

    stored = collection.get(
        include=[
            "documents",
            "metadatas",
            "embeddings"
        ]
    )

    documents = stored[
        "documents"
    ]

    metadatas = stored[
        "metadatas"
    ]

    if not documents:

        raise HTTPException(
            status_code=404,
            detail=(
                "No research content exists "
                "in this session."
            )
        )


    # ========================================================
    # DETERMINE PAPERS IN SESSION
    # ========================================================

    session_papers = []
    seen_papers = set()

    for metadata in metadatas:

        paper_name = metadata[
            "paper"
        ]

        if paper_name not in seen_papers:

            seen_papers.add(
                paper_name
            )

            session_papers.append(
                paper_name
            )

    comparison_papers = detect_comparison_papers(
        request.question,
        session_papers
    )

    is_comparison = (
        comparison_papers is not None
        and len(comparison_papers) >= 2
    )

    # Fail closed for unresolved explicit comparison requests.
    # Never fall back to normal session-wide retrieval, because that
    # can silently substitute a different paper (for example,
    # YouStruQ) for a paper the user explicitly requested.
    if (
        not is_comparison
        and comparison_papers is None
        and len(session_papers) >= 2
    ):
        comparison_intent_terms = [
            "compare",
            "comparison",
            "difference",
            "differences",
            "differ",
            "versus",
            "vs",
            "between",
            "how do they",
            "how are they",
            "different approaches",
            "different methods"
        ]

        if any(
            term in request.question.lower()
            for term in comparison_intent_terms
        ):
            raise HTTPException(
                status_code=400,
                detail=(
                    "I detected a comparison question, but I could not "
                    "match at least two requested paper names to the "
                    f"papers in this session: {', '.join(session_papers)}. "
                    "Please use the uploaded paper names in the question."
                )
            )

    paper_list_text = ", ".join(
        session_papers
    )


    # ========================================================
    # 1. QUERY EMBEDDING
    # ========================================================

    query_embedding = (
        embedding_model.encode(
            [request.question],
            normalize_embeddings=True
        )
    )

    # ========================================================
    # 2. BM25 INDEX
    # ========================================================

    tokenized_documents = [
        tokenize(document)
        for document in documents
    ]

    bm25 = BM25Okapi(
        tokenized_documents
    )

    query_tokens = tokenize(
        request.question
    )

    bm25_scores = bm25.get_scores(
        query_tokens
    )

    if is_comparison:

        reranked_results = retrieve_comparison_results(
            question=request.question,
            query_embedding=query_embedding,
            documents=documents,
            metadatas=metadatas,
            collection=collection,
            bm25_scores=bm25_scores,
            comparison_papers=comparison_papers
        )

    else:


        # ----------------------------------------------------
        # Semantic top 10 across the session
        # ----------------------------------------------------

        semantic_results = collection.query(
            query_embeddings=(
                query_embedding.tolist()
            ),
            n_results=min(
                10,
                len(documents)
            )
        )

        semantic_documents = (
            semantic_results[
                "documents"
            ][0]
        )

        semantic_metadatas = (
            semantic_results[
                "metadatas"
            ][0]
        )

        # ----------------------------------------------------
        # BM25 top 10 across the session
        # ----------------------------------------------------

        ranked_bm25_indices = sorted(
            range(len(bm25_scores)),
            key=lambda i: bm25_scores[i],
            reverse=True
        )[:10]

        bm25_documents = [
            documents[i]
            for i in ranked_bm25_indices
        ]

        bm25_metadatas = [
            metadatas[i]
            for i in ranked_bm25_indices
        ]

        # ========================================================
        # 3. RECIPROCAL RANK FUSION
        # ========================================================

        rrf_scores = {}

        k = 60

        # --------------------------------------------------------
        # Semantic ranking
        # --------------------------------------------------------

        for rank, metadata in enumerate(
            semantic_metadatas
        ):

            chunk_id = metadata[
                "chunk_id"
            ]

            if chunk_id not in rrf_scores:

                rrf_scores[
                    chunk_id
                ] = {

                    "score": 0,

                    "metadata": metadata,

                    "document":
                        semantic_documents[
                            rank
                        ]
                }

            rrf_scores[
                chunk_id
            ]["score"] += (
                1 / (
                    k + rank + 1
                )
            )

        # --------------------------------------------------------
        # BM25 ranking
        # --------------------------------------------------------

        for rank, metadata in enumerate(
            bm25_metadatas
        ):

            chunk_id = metadata[
                "chunk_id"
            ]

            if chunk_id not in rrf_scores:

                rrf_scores[
                    chunk_id
                ] = {

                    "score": 0,

                    "metadata": metadata,

                    "document":
                        bm25_documents[
                            rank
                        ]
                }

            rrf_scores[
                chunk_id
            ]["score"] += (
                1 / (
                    k + rank + 1
                )
            )

        # --------------------------------------------------------
        # Sort fused results
        # --------------------------------------------------------

        fused_results = sorted(

            rrf_scores.values(),

            key=lambda x:
                x["score"],

            reverse=True

        )[:10]

        # ========================================================
        # 4. CROSS-ENCODER RERANKING
        # ========================================================

        rerank_pairs = [

            (
                request.question,
                result["document"]
            )

            for result in fused_results
        ]

        if rerank_pairs:

            rerank_scores = (
                cross_encoder.predict(
                    rerank_pairs
                )
            )

            for result, score in zip(
                fused_results,
                rerank_scores
            ):

                result[
                    "rerank_score"
                ] = float(score)

            reranked_results = sorted(

                fused_results,

                key=lambda x:
                    x["rerank_score"],

                reverse=True

            )[:5]

        else:

            reranked_results = []

    # ========================================================
    # 5. BUILD EVIDENCE
    # ========================================================

    evidence_blocks = []

    sources = []

    if is_comparison:

        # Keep comparison evidence visibly separated by paper.
        for paper_name in comparison_papers:

            paper_results = [
                result
                for result in reranked_results
                if result["metadata"]["paper"] == paper_name
            ]

            if not paper_results:
                continue

            evidence_blocks.append(
                f"=== {paper_name} ==="
            )

            for index, result in enumerate(
                paper_results,
                start=1
            ):

                metadata = result[
                    "metadata"
                ]

                page_number = metadata[
                    "page"
                ]

                passage = result[
                    "document"
                ]

                prompt_passage = passage
                if len(prompt_passage) > 3200:
                    prompt_passage = (
                        prompt_passage[:3200].rsplit(" ", 1)[0]
                        + " ..."
                    )

                evidence_blocks.append(
                    f"""
EVIDENCE {index}
Paper: {paper_name}
Page: {page_number}
Passage:
{prompt_passage}
""".strip()
                )

                sources.append(
                    EvidenceSource(
                        paper_name=paper_name,
                        page_number=page_number,
                        passage=passage
                    )
                )

    else:

        for index, result in enumerate(
            reranked_results,
            start=1
        ):

            metadata = result[
                "metadata"
            ]

            paper_name = metadata[
                "paper"
            ]

            page_number = metadata[
                "page"
            ]

            passage = result[
                "document"
            ]

            evidence_blocks.append(
                f"""
EVIDENCE {index}
Paper: {paper_name}
Page: {page_number}
Passage:
{passage}
""".strip()
            )

            sources.append(
                EvidenceSource(
                    paper_name=paper_name,
                    page_number=page_number,
                    passage=passage
                )
            )

    evidence_text = (
        "\n\n".join(
            evidence_blocks
        )
    )


    if is_comparison:

        comparison_instructions = """
COMPARISON MODE IS ACTIVE.

The user is asking about multiple papers. Evidence has been
retrieved separately for each identified paper.

- Compare ONLY the exact requested papers listed in the
  REQUESTED COMPARISON PAPERS section below.
- Do not introduce, substitute, or compare any other paper from the session.
- The paper name in the evidence heading is authoritative.
- A task, attack scenario, dataset, model, subsection, or example
  mentioned inside a paper is NOT a separate paper.
- Evidence is retrieved separately for each identified paper.
- Keep every claim attached to the correct paper.
- Do not let evidence from one paper substitute for missing evidence from another.
- Organize the answer by useful comparison dimensions such as
  objective, problem, method, evaluation, or contribution.
- Explicitly name the paper when describing a difference.
- If one paper lacks evidence for a comparison dimension, say so
  rather than filling the gap from general knowledge.
- If BOTH requested papers have substantive evidence, you MUST provide
  a direct comparison. Do not use the generic "not enough information"
  refusal merely because one particular dimension is missing.
- Prefer evidence from the paper's abstract/introduction/method/evaluation
  over an isolated attack example when describing the paper's overall focus.
- Do not treat an example or scenario inside a paper as the paper's identity.

MARKDOWN TABLE RULES:
- If you use a comparison table, it MUST be valid Markdown.
- Use exactly one header row and one separator row.
- Put a pipe (|) between every cell.
- Use this structure:

| Dimension | Paper A | Paper B |
|---|---|---|
| Objective | ... | ... |
| Problem | ... | ... |
| Method | ... | ... |
| Evaluation | ... | ... |
| Contribution | ... | ... |

- Replace Paper A and Paper B with the actual paper names.
- Do not concatenate the header text with the first cell.
- Do not output HTML tables.
- After the table, give a short 2-4 sentence synthesis of the main differences.
- Prefer concrete details from the evidence (tasks, framework, evaluation setup, metrics, or contributions) over generic wording.
"""

    else:

        comparison_instructions = ""

    # ========================================================
    # 7. LLM ANSWER GENERATION
    # ========================================================

    comparison_papers_text = ", ".join(
        comparison_papers
    ) if is_comparison else ""

    prompt = f"""
You are the answer engine for ResearchVault,
an AI research assistant.

The user has uploaded one or more research papers
into the current research session.

PAPERS IN THIS SESSION:
{paper_list_text}

Your task is to answer the user's question using
ONLY the retrieved evidence.

USER QUESTION:
{request.question}

REQUESTED COMPARISON PAPERS:
{comparison_papers_text if is_comparison else "None"}

SOURCE CONTRACT:
{(
    "This is an explicit paper comparison. Compare ONLY these requested papers: "
    + comparison_papers_text
    + ". Every evidence block is labeled with its source paper. "
    "Both requested papers have passed the provenance check."
    if is_comparison
    else
    "This is not an explicit paper comparison."
)}

RETRIEVED EVIDENCE:
{evidence_text}

{comparison_instructions}

FOLLOW THESE RULES CAREFULLY:

1. Answer the user's question directly.

2. Use ONLY information supported by the retrieved
   evidence.

3. Do NOT use your general knowledge to fill missing
   information.

4. If the evidence explicitly answers the question,
   explain the answer clearly and confidently.

5. If the evidence provides only partial information,
   explain what the evidence supports and clearly
   state what is not established.

6. If the evidence is only partial, answer the parts
   that ARE supported and identify only the specific
   missing parts. Do not turn a partially supported
   comparison into a full generic refusal.

   For an explicit comparison, if there is substantive
   evidence for BOTH requested papers, answer the comparison
   directly even if some dimensions are not covered.

   Only use the following sentence when essentially no
   useful evidence for the requested comparison is present:

   "The uploaded papers do not explicitly provide
   enough information to answer this question."

   Then briefly explain what related information
   IS present in the retrieved evidence, if useful.

7. Do NOT invent definitions for terms that are not
   explicitly defined.

8. Pay close attention to acronyms, project names,
   model names, people, organizations, and
   technical terms.

9. Treat every uploaded PDF as ONE research paper.
   Do not turn section headings, subsections,
   experiments, datasets, models, or components
   into separate papers.

10. Preserve the exact terminology used by each paper.

11. When multiple papers are relevant, clearly
    distinguish which information belongs to which
    paper.

12. If the question asks to compare papers, organize
    the answer by paper or comparison dimension.

13. If the question asks "What is X?", first determine
    whether the retrieved evidence actually defines X.
    If it does not, do not invent a definition.

14. Prefer a concise, useful answer over a generic
    refusal.

15. Do not mention retrieval, embeddings, BM25,
    cross-encoders, ChromaDB, or internal system
    implementation details.

16. Do not create fake citations or page numbers.
    The application provides source information
    separately.

17. Never merge facts from two different papers as
    though they came from the same paper.

18. When answering a multi-paper question, explicitly
    identify the paper when the source of a claim
    matters.

19. For comparison questions, make sure EACH requested
    paper is represented in the answer. Never substitute another
    uploaded paper. If a comparison dimension is not supported for
    one paper, write "Not established by the retrieved evidence"
    instead of guessing.

20. A passage describing an attack, task, dataset, model, or
    example from AgentDojo is still part of AgentDojo; it cannot
    be treated as a second paper. The same rule applies to every
    uploaded paper.

21. If you output a Markdown comparison table, verify
    that every row has the same number of cells and that
    the separator row is valid Markdown.

22. If only one paper is supported by the retrieved
    evidence, do not pretend that the other paper
    contains the same information.

23. Answer only from the supplied evidence.

Now answer the question.
"""

    # ========================================================
    # OLLAMA
    # ========================================================

    try:

        response = ollama.chat(

            model=LLM_MODEL_NAME,

            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        answer = (
            response[
                "message"
            ][
                "content"
            ]
            .strip()
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                f"LLM generation failed: {exc}"
            )
        )

    # ========================================================
    # 8. RETURN ANSWER + SOURCES
    # ========================================================

    return AnswerResponse(

        answer=answer,

        sources=sources
    )


# ========================================================
# END SESSION
# ========================================================

@app.delete(
    "/session/{session_id}"
)
def end_session(
    session_id: str
):

    collection_name = (
        get_collection_name(
            session_id
        )
    )

    session_dir = (
        TEMP_STORAGE_DIR /
        session_id
    )

    # --------------------------------------------------------
    # Delete Chroma collection
    # --------------------------------------------------------

    try:

        chroma_client.delete_collection(
            name=collection_name
        )

    except Exception:

        pass

    # --------------------------------------------------------
    # Delete temporary PDFs
    # --------------------------------------------------------

    if session_dir.exists():

        shutil.rmtree(
            session_dir,
            ignore_errors=True
        )

    return {

        "message":
            "Research session ended successfully.",

        "session_id":
            session_id
    }