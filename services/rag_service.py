"""
services/rag_service.py

The "librarian" of the whole system. It knows how to:
  1. Read your notes (PDF, Word, plain text) from the data/ folder
  2. Chop them into small overlapping chunks
  3. Convert each chunk into a vector (numbers that capture its MEANING)
  4. Store those vectors in Pinecone, a managed cloud vector database
  5. Given a new question, find the chunks whose meaning is closest to it

This is what RAG (Retrieval-Augmented Generation) means: instead of the LLM
guessing an answer from what it memorized during training, we hand it the
actual relevant passages from YOUR notes first.

Using Pinecone instead of local Chroma means:
  - Vectors persist across redeploys (no re-ingestion on every cold start)
  - No disk storage needed on the host
  - Scales automatically
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import Pinecone as PineconeVectorStore
from pinecone import Pinecone as PineconeClient

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

# Which loader to use, based on file extension
LOADER_MAP = {
    ".pdf": PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".txt": TextLoader,
}

PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "learning-notes")


class RAGService:
    def __init__(self):
        # Local embedding model — runs on the server, free, no API key.
        # First run downloads it once (~90MB), then it's cached locally.
        # Produces 384-dimensional vectors — must match the Pinecone index
        # dimension set during index creation.
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Pinecone client — reads PINECONE_API_KEY from env automatically.
        pc = PineconeClient(api_key=os.getenv("PINECONE_API_KEY"))
        index = pc.Index(PINECONE_INDEX_NAME)

        # LangChain wrapper around the Pinecone index. Uses the same
        # .add_documents() / .similarity_search() interface as Chroma, so
        # nothing else in the codebase needs to change.
        self.vector_store = PineconeVectorStore(
            index=index,
            embedding=self.embeddings,
            namespace="learning_notes",
        )

        # Keep a direct reference to the Pinecone index for count().
        self._index = index

    def ingest_folder(self, folder: str = str(DATA_DIR)) -> int:
        """
        Reads every supported file in `folder`, splits it into chunks,
        and upserts those chunks into Pinecone.

        Because Pinecone is persistent, this only needs to run once (or
        when you add new notes). The lifespan handler in main.py calls
        this only when the index is empty, so you won't get duplicates
        on normal redeploys.

        Returns the number of chunks added.
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
        )

        all_chunks = []
        for path in Path(folder).iterdir():
            if not path.is_file():
                continue

            loader_cls = LOADER_MAP.get(path.suffix.lower())
            if loader_cls is None:
                continue  # skip files we don't know how to read

            docs = loader_cls(str(path)).load()
            chunks = splitter.split_documents(docs)

            # Tag each chunk with its source file, so later we can tell
            # the user WHERE an answer came from.
            for chunk in chunks:
                chunk.metadata["source"] = path.name

            all_chunks.extend(chunks)

        if all_chunks:
            self.vector_store.add_documents(all_chunks)

        return len(all_chunks)

    def retrieve(self, query: str, k: int = 4):
        """
        Given a question, return the k chunks from your notes whose
        MEANING is closest to it (not just keyword matching).

        Each result has .page_content (the text) and .metadata (source file).
        """
        return self.vector_store.similarity_search(query, k=k)

    def count(self) -> int:
        """
        How many vectors are currently in the Pinecone index namespace.
        Used at startup to decide whether we need to (re-)ingest data/ --
        with Pinecone, data persists across deploys so this will usually
        be non-zero after the first run.

        NOTE: pinecone v9 returns a DescribeIndexStatsResponse *object*,
        not a plain dict, so we use attribute access, not .get().
        """
        stats = self._index.describe_index_stats()
        # stats.namespaces is dict[str, NamespaceDescription]; each entry
        # has a .vector_count attribute.
        ns = stats.namespaces.get("learning_notes")
        return ns.vector_count if ns is not None else 0
