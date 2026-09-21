import chromadb


def create_collection():

    client = chromadb.PersistentClient(
        path="data/chroma"
    )

    collection = client.get_or_create_collection(
        name="research_papers"
    )

    return collection


def add_documents(
    collection,
    chunks,
    embeddings
):

    ids = []
    documents = []
    metadatas = []

    for chunk, embedding in zip(
        chunks,
        embeddings
    ):

        chunk_id = chunk["chunk_id"]

        ids.append(chunk_id)

        documents.append(
            chunk["text"]
        )

        metadatas.append({

            "paper": chunk["paper"],

            "page": chunk["page"],

            "chunk_id": chunk_id

        })

    collection.add(

        ids=ids,

        documents=documents,

        embeddings=embeddings.tolist(),

        metadatas=metadatas

    )


if __name__ == "__main__":

    collection = create_collection()

    print(
        "ChromaDB collection ready!"
    )

    print(
        "Collection name:",
        collection.name
    )