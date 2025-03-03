import os
import re
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import DirectoryLoader, Docx2txtLoader

# Define file paths
FAISS_INDEX_PATH = "index.faiss"  # FAISS binary index
FAISS_METADATA_PATH = "index.pkl"  # Metadata file

# Initialize embedding model
MODEL = SentenceTransformer('all-MiniLM-L6-v2')

def load_and_clean_documents(folder_path):
    """Load and clean .docx documents from a folder."""
    txt_loader = DirectoryLoader(folder_path, glob="*.docx", loader_cls=Docx2txtLoader)
    raw_documents = txt_loader.load()

    cleaned_docs = []
    for doc in raw_documents:
        text = re.sub(r'\s+', ' ', doc.page_content).strip()
        doc.metadata['source'] = doc.metadata.get('source', f"doc_{len(cleaned_docs) + 1}")  # Ensure unique source
        cleaned_docs.append({'text': text, 'metadata': doc.metadata})

    return cleaned_docs

def load_existing_data():
    """Load existing FAISS index and metadata if available."""
    if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(FAISS_METADATA_PATH):
        print("✅ Loading existing FAISS index...")
        index = faiss.read_index(FAISS_INDEX_PATH)
        with open(FAISS_METADATA_PATH, "rb") as f:
            metadata = pickle.load(f)
        print(f"🔢 FAISS index contains {index.ntotal} embeddings.")
    else:
        print("🆕 No existing FAISS index found. Creating a new one.")
        index = faiss.IndexFlatL2(384)  # Create new index
        metadata = []
    return index, metadata

def generate_embeddings_and_save(cleaned_docs):
    """Generate embeddings and store them in FAISS only if they are new."""
    index, metadata = load_existing_data()
    existing_sources = {doc['source'] for doc in metadata}
    
    new_docs = [doc for doc in cleaned_docs if doc['metadata']['source'] not in existing_sources]
    
    if not new_docs:
        print("⚠️ No new documents to add. FAISS index remains unchanged.")
        return
    
    for doc in new_docs:
        print(f"📄 Generating embedding for {doc['metadata']['source']}...")
        embedding = MODEL.encode([doc['text']])[0].astype('float32')
        index.add(np.expand_dims(embedding, axis=0))
        metadata.append({'source': doc['metadata']['source'], 'text': doc['text']})
    
    print("💾 Saving updated FAISS index and metadata...")
    faiss.write_index(index, FAISS_INDEX_PATH)
    with open(FAISS_METADATA_PATH, "wb") as f:
        pickle.dump(metadata, f)
    
    print(f"✅ Added {len(new_docs)} new documents. Total embeddings now: {index.ntotal}")

def check_stored_embeddings():
    """Check if embeddings are stored in FAISS."""
    if not os.path.exists(FAISS_INDEX_PATH):
        print("❌ FAISS index file not found. No embeddings stored.")
        return

    index = faiss.read_index(FAISS_INDEX_PATH)
    total_embeddings = index.ntotal

    print(f"📊 Total stored embeddings: {total_embeddings}")

    if total_embeddings > 0:
        example_vector = np.zeros((1, index.d), dtype=np.float32)
        try:
            index.reconstruct(0, example_vector[0])
            print(f"🔎 Example stored embedding (first 10 values): {example_vector[0][:10]}")
        except RuntimeError:
            print("⚠️ FAISS index is empty. No embeddings to reconstruct.")
    else:
        print("⚠️ No embeddings found in FAISS index.")

if __name__ == "__main__":
    folder_path = "text_files"
    
    print("📂 Loading and cleaning documents...")
    cleaned_docs = load_and_clean_documents(folder_path)
    
    print("\n📥 Generating and storing embeddings in FAISS...")
    generate_embeddings_and_save(cleaned_docs)
    
    print("\n🔎 Checking stored embeddings in FAISS...")
    check_stored_embeddings()
