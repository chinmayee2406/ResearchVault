import pymupdf
from pathlib import Path


PAPERS_DIR = Path("data/papers")


def load_papers():

    papers = []

    for pdf_path in PAPERS_DIR.glob("*.pdf"):

        document = pymupdf.open(pdf_path)

        paper_name = pdf_path.stem

        for page_number, page in enumerate(document):

            text = page.get_text().strip()

            if not text:
                continue

            papers.append({
                "paper": paper_name,
                "page": page_number + 1,
                "text": text
            })

    return papers


if __name__ == "__main__":

    papers = load_papers()

    print("Total pages loaded:", len(papers))

    print("\nPapers found:")

    paper_names = sorted(
        set(page["paper"] for page in papers)
    )

    for paper in paper_names:
        print("-", paper)

    print("\nFirst page of each paper:")

    for paper in paper_names:

        first_page = next(
            page for page in papers
            if page["paper"] == paper
        )

        print("\n" + "=" * 60)
        print("Paper:", first_page["paper"])
        print("Page:", first_page["page"])
        print("Text:", first_page["text"][:300])