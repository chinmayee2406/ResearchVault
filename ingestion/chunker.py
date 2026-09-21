from pathlib import Path

import pymupdf


PAPERS_DIR = Path("data/papers")

CHUNK_SIZE = 500
OVERLAP = 50

REFERENCE_START_PAGES = {
    "attention": 8,
    "RAG": 17
}


def is_reference_page(text):

    text_lower = text.lower()

    beginning = text_lower[:500]

    if "references" in beginning:
        return True

    if "bibliography" in beginning:
        return True

    citation_count = 0

    for i in range(1, 6):

        if f"[{i}]" in beginning:

            citation_count += 1

    if citation_count >= 3:
        return True

    return False


def load_chunks():

    all_chunks = []

    for pdf_path in PAPERS_DIR.glob("*.pdf"):

        document = pymupdf.open(pdf_path)

        paper_name = pdf_path.stem

        paper_chunk_id = 0

        for page_number, page in enumerate(document):

            text = page.get_text().strip()

            if not text:
                continue

            reference_start = (
                REFERENCE_START_PAGES.get(
                    paper_name
                )
            )

            if (
                reference_start is not None
                and page_number + 1 >= reference_start
            ):

                print(
                    f"Skipping reference page: "
                    f"{paper_name} - Page "
                    f"{page_number + 1}"
                )

                continue

            if is_reference_page(text):

                print(
                    f"Skipping reference page: "
                    f"{paper_name} - Page "
                    f"{page_number + 1}"
                )

                continue

            words = text.split()

            step = CHUNK_SIZE - OVERLAP

            for i in range(
                0,
                len(words),
                step
            ):

                chunk = words[
                    i:i + CHUNK_SIZE
                ]

                if chunk:

                    all_chunks.append({

                        "paper": paper_name,

                        "page": page_number + 1,

                        "chunk_id": (
                            f"{paper_name}_chunk_"
                            f"{paper_chunk_id}"
                        ),

                        "text": " ".join(chunk)

                    })

                    paper_chunk_id += 1

    return all_chunks


if __name__ == "__main__":

    chunks = load_chunks()

    print(
        "\nTotal chunks:",
        len(chunks)
    )

    print(
        "\nChunks by paper:"
    )

    paper_counts = {}

    for chunk in chunks:

        paper = chunk["paper"]

        paper_counts[paper] = (
            paper_counts.get(
                paper,
                0
            ) + 1
        )

    for paper, count in paper_counts.items():

        print(
            f"- {paper}: {count}"
        )