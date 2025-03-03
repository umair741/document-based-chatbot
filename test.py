import os
import pickle
import faiss
from langchain_community.vectorstores import FAISS as LC_FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain_community.docstore import InMemoryDocstore
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# FAISS Index Directory
FAISS_INDEX_DIR = r"D:\chatsbot"  # Change this to your correct directory
FAISS_INDEX_PATH = os.path.join(FAISS_INDEX_DIR, "index.faiss")
FAISS_METADATA_PATH = os.path.join(FAISS_INDEX_DIR, "index.pkl")

# Load embedding model
MODEL_NAME = 'all-MiniLM-L6-v2'
embedding_model = HuggingFaceEmbeddings(model_name=f"sentence-transformers/{MODEL_NAME}")

def check_faiss_files():
    """Check if FAISS index and metadata exist."""
    print("🔍 Checking FAISS files...")
    if not os.path.exists(FAISS_INDEX_PATH):
        print("❌ FAISS index file is missing!")
    else:
        print("✅ FAISS index file exists.")

    if not os.path.exists(FAISS_METADATA_PATH):
        print("❌ FAISS metadata file is missing!")
    else:
        print("✅ FAISS metadata file exists.")

def load_faiss_vectorstore():
    """Load FAISS index and metadata, checking for issues."""
    if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(FAISS_METADATA_PATH):
        print("❌ FAISS index or metadata file is missing!")
        return None

    raw_index = faiss.read_index(FAISS_INDEX_PATH)
    with open(FAISS_METADATA_PATH, "rb") as f:
        metadata_list = pickle.load(f)

    print(f"✅ FAISS index loaded with {raw_index.ntotal} embeddings.")

    if raw_index.ntotal == 0:
        print("⚠️ FAISS index is empty! No documents available for retrieval.")
        return None

    # Convert metadata into Document objects
    docstore = InMemoryDocstore({
        i: Document(page_content=entry.get("text", ""), metadata={"source": entry.get("source", "unknown")})
        for i, entry in enumerate(metadata_list)
    })

    # Create FAISS vector store
    vector_store = LC_FAISS(
        embedding_function=embedding_model,
        index=raw_index,
        docstore=docstore,
        index_to_docstore_id={i: i for i in range(len(metadata_list))}
    )

    return vector_store

def test_retrieval(vector_store):
    """Test retrieval by running a sample query."""
    if not vector_store:
        print("❌ Cannot test retrieval - FAISS is not loaded.")
        return
    
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    test_query = "What is Machine Learning?"
    retrieved_docs = retriever.get_relevant_documents(test_query)
    
    print(f"\n🔎 Retrieval test for query: '{test_query}'")
    if not retrieved_docs:
        print("⚠️ No documents retrieved! Check FAISS indexing.")
    else:
        for i, doc in enumerate(retrieved_docs):
            print(f"📄 Document {i+1}: {doc.page_content[:300]}...")

def run_debug_checks():
    """Run all debugging checks."""
    check_faiss_files()
    vector_store = load_faiss_vectorstore()
    test_retrieval(vector_store)

if __name__ == "__main__":
    run_debug_checks()
