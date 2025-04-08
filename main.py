import os
import pickle
import faiss
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS as LC_FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain.schema import Document

# Load environment variables
load_dotenv()

# File paths for the FAISS index and metadata
FAISS_INDEX_PATH = "index.faiss"
FAISS_METADATA_PATH = "index.pkl"

# Global chatbot variable
chatbot = None

# Setup the embedding model
MODEL_NAME = 'all-MiniLM-L6-v2'
embedding_model = HuggingFaceEmbeddings(model_name=f"sentence-transformers/{MODEL_NAME}")
MODEL = SentenceTransformer(f"sentence-transformers/{MODEL_NAME}")

def verify_documents():
    """Prints out the stored documents for verification."""
    try:
        with open(FAISS_METADATA_PATH, "rb") as f:
            metadata = pickle.load(f)
    except Exception as e:
        print("Error loading metadata:", e)
        return

    print("\nStored Documents:")
    for i, doc in enumerate(metadata):
        print(f"\nDocument {i+1}: {doc['source']}")
        print("First 100 chars:", doc['text'][:100])

class SimpleDocstore:
    def __init__(self, store: dict):
        self.store = store

    def search(self, doc_id):
        return self.store.get(doc_id, None)

def load_faiss_vectorstore():
    """Loads the FAISS index and metadata, and returns a vector store instance."""
    if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(FAISS_METADATA_PATH):
        raise FileNotFoundError("❌ FAISS index or metadata file is missing!")

    # Load the FAISS index
    raw_index = faiss.read_index(FAISS_INDEX_PATH)

    # Load metadata
    with open(FAISS_METADATA_PATH, "rb") as f:
        metadata_list = pickle.load(f)

    # Convert metadata entries into Document objects
    docstore_dict = {
        i: Document(page_content=entry.get("text", ""), metadata={"source": entry.get("source", "unknown")})
        for i, entry in enumerate(metadata_list)
    }
    docstore = SimpleDocstore(docstore_dict)
    index_to_docstore_id = {i: i for i in range(len(metadata_list))}

    vector_store = LC_FAISS(
        embedding_function=embedding_model,
        index=raw_index,
        index_to_docstore_id=index_to_docstore_id,
        docstore=docstore
    )
    print(f"✅ FAISS vector store loaded with {raw_index.ntotal} embeddings.")
    return vector_store

def initialize_chatbot():
    """Initializes the chatbot using the latest FAISS index and returns the chatbot instance."""
    global chatbot
    vector_store = load_faiss_vectorstore()
    
    # Verify documents in the index
    verify_documents()
    
    # Initialize the Gemini LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp",
        temperature=0.7,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    # Define a prompt template that enforces document-based responses
    prompt_template = PromptTemplate(
        input_variables=["context", "chat_history", "question"],
        template=(
            "You are an AI assistant that answers questions strictly based on the provided documents.\n"
            "If the answer is not found in the documents, respond with:\n"
            "Always respond in English, regardless of the input language.\n"
            "'I don't have an answer based on the provided documents.'\n\n"
            "If the question mentions explain  in Sindhi or Urdu, respond in the same language urdu or sindh. Otherwise, respond in English.\n"
            "Documents:\n{context}\n\n"
            "Chat History:\n{chat_history}\n"
            "User Question: {question}\n\n"
            "Answer:"
        )
    )

    # Setup memory for conversation history
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

    # Build the conversational retrieval chain (chatbot)
    chatbot = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt_template},
        return_source_documents=False
    )
    print("Chatbot initialized successfully!")
    return chatbot

def get_chatbot_response(user_input):
    """Returns the chatbot's response for the given user input."""
    global chatbot
    if chatbot is None:
        return "⚠️ Chatbot is not initialized."
    try:
        response = chatbot.invoke({"question": user_input})
        return response["answer"]
    except Exception as e:
        return f"❌ Error: {str(e)}"

if __name__ == "__main__":
    # Initialize the chatbot when running the script directly.
    initialize_chatbot()
    print("Chatbot is ready! Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting chatbot. Goodbye!")
            break
        answer = get_chatbot_response(user_input)
        print("AIBot:", answer)
