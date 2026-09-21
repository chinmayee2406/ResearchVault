from pathlib import Path

from fastapi import APIRouter, File, UploadFile

from sentence_transformers import SentenceTransformer

from ingestion.chunker import load_chunks
from embeddings.vector_store import (
    create_collection,
    add_documents
)


PAPERS_DIR = Path("data/papers")

router = APIRouter()

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


@router.post("/upload")
async def upload_paper(
    file: UploadFile = File(...)
):

    if not file.filename.lower().endswith(".pdf"):

        return {
            "error": "Only PDF files are supported."
        }

    PAPERS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    file_path = PAPERS_DIR / file.filename

    file_content = await file.read()

    with open(file_path, "wb") as output_file:

        output_file.write(file_content)

    chunks = load_chunks()

    new_chunks = [
        chunk
        for chunk in chunks
        if chunk["paper"] == file_path.stem
    ]

    if not new_chunks:

        return {
            "error": "No text could be extracted from the PDF."
        }

    collection = create_collection()

    existing = collection.get()

    existing_chunk_ids = set(
        existing["ids"]
    )

    filtered_chunks = []

    for chunk in new_chunks:

        chunk_id = f"chunk_{chunk['chunk_id']}"

        if chunk_id not in existing_chunk_ids:

            filtered_chunks.append(chunk)

    if not filtered_chunks:

        return {
            "message": "Paper already exists.",
            "paper": file_path.stem,
            "chunks_added": 0
        }

    texts = [
        chunk["text"]
        for chunk in filtered_chunks
    ]

    embeddings = embedding_model.encode(
        texts
    )

    add_documents(
        collection,
        filtered_chunks,
        embeddings
    )

    return {
        "message": "Paper uploaded successfully.",
        "paper": file_path.stem,
        "chunks_added": len(filtered_chunks)
    }