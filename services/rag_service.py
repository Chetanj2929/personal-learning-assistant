"""
services/rag_service.py

The "librarian" of the whole system. It knows how to:
  1. Read your notes (PDF, Word, plain text) from the data/ folder
  2. Chop them into small overlapping chunks
  3. Convert each chunk into a vector (numbers that capture its MEANING)
  4. Store those vectors in Chroma, a local vector database
  5. Given a new question, find the chunks whose meaning is closest to it

This is what RAG (Retrieval-Augmented Generation) means: instead of the LLM
guessing an answer from what it memorized during training, we hand it the
actual relevant passages from YOUR notes first.
"""

from pathlib import Path

from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
VECTOR_DB_DIR = BASE_DIR / "data_vector_db"

# Which loader to use, based on file extension
LOADER_MAP = {
    ".pdf": PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".txt": TextLoader,
}


class RAGService:
    def __init__(self):
        # Local embedding model — runs on your machine, free, no API key.
        # First run downloads it once (~90MB), then it's cached locally.
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Chroma persists to disk, so notes you ingest stay available
        # across restarts — you don't have to re-ingest every time.
        self.vector_store = Chroma(
            collection_name="learning_notes",
            embedding_function=self.embeddings,
            persist_directory=str(VECTOR_DB_DIR),
        )

    def ingest_folder(self, folder: str = str(DATA_DIR)) -> int:
        """
        Reads every supported file in `folder`, splits it into chunks,
        and adds those chunks to the vector store.

        Safe to re-run whenever you add new notes — it just adds more
        chunks on top (doesn't currently de-duplicate re-ingested files,
        worth knowing if you run this twice on the same file).

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
        How many chunks are currently in the vector store. Used at startup
        to decide whether we need to (re-)ingest data/ -- relevant on hosts
        with EPHEMERAL storage (like Render's free tier), where the vector
        store resets to empty on every fresh deploy.
        """
        return self.vector_store._collection.count()
