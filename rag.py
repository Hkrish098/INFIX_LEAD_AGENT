from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma(
    persist_directory="vector_db",
    embedding_function=embeddings
)

def retrieve_answer(query: str) -> str:
    docs = vectorstore.similarity_search(query, k=3)
    return "\n".join(d.page_content for d in docs)
