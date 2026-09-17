import pymupdf
from sentence_transformers import SentenceTransformer

pdf_path = "data/papers/attention.pdf"

document = pymupdf.open(pdf_path)

chunk_size = 500
overlap = 50

all_chunks = []

for page_number, page in enumerate(document):

    text = page.get_text().strip()

    if not text:
        continue

    words = text.split()

    step = chunk_size - overlap

    for i in range(0, len(words), step):

        chunk = words[i:i + chunk_size]

        if chunk:

            chunk_data = {
                "paper": "NeMo Guardrails",
                "page": page_number + 1,
                "chunk_id": len(all_chunks),
                "text": " ".join(chunk)
            }

            all_chunks.append(chunk_data)


print("Total chunks:", len(all_chunks))


model = SentenceTransformer("all-MiniLM-L6-v2")

texts = [chunk["text"] for chunk in all_chunks]

embeddings = model.encode(texts)

print("Embeddings generated!")
print("Number of embeddings:", len(embeddings))
print("Dimensions per embedding:", len(embeddings[0]))