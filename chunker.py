import pymupdf
pdf_path = "data/papers/attention.pdf"
document = pymupdf.open(pdf_path)
chunk_size=500
overlap=50
all_chunks=[]
for page_number,page in enumerate(document):
    text=page.get_text().strip()
    if not text:
        continue
    words = text.split()
    step = chunk_size-overlap
    for i in range(0,len(words),step):
        chunk = words[i:i+chunk_size]
        if chunk:
            chunk_data={
                "paper": "NeMo Guardrails",
                "page": page_number + 1,
                "chunk_id": len(all_chunks),
                "text": " ".join(chunk)
            }
            all_chunks.append(chunk_data)
print("Total chunks:", len(all_chunks))
print("\nFirst 3 chunks:")
for chunk in all_chunks[:3]:
    print("\n" + "=" * 60)
    print("Chunk ID:", chunk["chunk_id"])
    print("Paper:", chunk["paper"])
    print("Page:", chunk["page"])
    print("Text:", chunk["text"][:300])
    print("\nChunks per page:")
page_counts = {}

for chunk in all_chunks:

    page = chunk["page"]

    if page not in page_counts:
        page_counts[page] = 0

    page_counts[page] += 1

for page, count in page_counts.items():

    print(f"Page {page}: {count} chunks")