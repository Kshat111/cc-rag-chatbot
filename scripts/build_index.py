"""Indexing pipeline entry point: ingest -> chunk -> build dense + BM25 indexes."""

from ragbot import chunking, index, ingest


def main():
    docs = ingest.load_documents()
    chunks = chunking.chunk_documents(docs)
    index.build_dense_index(chunks)
    index.build_bm25_index(chunks)


if __name__ == "__main__":
    main()
