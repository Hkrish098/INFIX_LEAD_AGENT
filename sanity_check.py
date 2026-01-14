from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VECTOR_DB_PATH = os.path.join(BASE_DIR, "vector_db")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = Chroma(
    persist_directory=VECTOR_DB_PATH,
    embedding_function=embeddings
)

docs = db.similarity_search("pricing plans", k=3)

print("Documents found:", len(docs))
print("-" * 40)

for i, d in enumerate(docs, 1):
    print(f"[{i}]")
    print(d.page_content[:300])
    print("-" * 40)
