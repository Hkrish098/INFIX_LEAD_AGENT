# scripts/ingest.py

import os
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(PROJECT_ROOT, "data")
VECTOR_DB_PATH = os.path.join(PROJECT_ROOT, "vector_db")

def ingest():
    docs = []

    # Load Markdown knowledge
    md_loader = TextLoader(
        os.path.join(DATA_PATH, "manuals", "autostream_info.md"),
        encoding="utf-8"
    )
    docs.extend(md_loader.load())

    # Load FAQs CSV (optional, if you want)
    faq_path = os.path.join(DATA_PATH, "faqs.csv")
    if os.path.exists(faq_path):
        with open(faq_path, "r", encoding="utf-8") as f:
            docs.append(Document(page_content=f.read()))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTOR_DB_PATH
    )

    print("✅ Vector DB ingested successfully")

if __name__ == "__main__":
    ingest()
