import pymupdf

pdf_path = "data/papers/attention.pdf"

document = pymupdf.open(pdf_path)

print("PDF opened successfully!")
print("Number of pages:", len(document))
pages=[]
for page_number, page in enumerate(document):
    text = page.get_text().strip()
    page_data = {
        "paper" : "NeMo Guardrails",
        "page":page_number+1,
        "text":text
    }
    pages.append(page_data)
print("Total pages stored:",len(pages))
print("\nFirst page data:")
print(pages[0])