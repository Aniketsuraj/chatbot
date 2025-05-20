import os
from datetime import datetime
import pandas as pd
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from langchain.embeddings import SentenceTransformerEmbeddings
from config.settings import VECTOR_STORE_DIR, EMBEDDING_MODEL

def get_vector_store_path(teacher_id):
    """Get the vector store path for a specific teacher."""
    return f'{VECTOR_STORE_DIR}/teacher_{teacher_id}'

def get_embeddings():
    """Get the embedding function for vector stores."""
    return SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)

def get_similar_conversations(question: str, teacher_id: str, k=1) -> list:
    """Find similar conversations from the vector store."""
    try:
        persist_directory = get_vector_store_path(teacher_id)
        embeddings = get_embeddings()
        
        if not os.path.exists(persist_directory):
            print(f"No vector store found for teacher {teacher_id}.")
            return []
            
        vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings
        )
        
        similar_docs = vector_store.similarity_search(
            query=question,
            k=k
        )
        
        similar_queries = [doc.metadata.get("sql_query", "N/A") for doc in similar_docs]
        
        return similar_queries
        
    except Exception as e:
        print(f"Error retrieving similar conversations: {e}")
        return []

def save_to_vector_store(
    question: str, 
    query: str = None, 
    response: str = None, 
    sql_result: pd.DataFrame = None, 
    tag: str = None,
    teacher_id: str = None,
    error_message: str = None
):
    """Save conversation data to vector store."""
    try:
        if not teacher_id:
            from streamlit import session_state
            teacher_id = session_state.get('teacher_id', 'default')
            
        persist_directory = get_vector_store_path(teacher_id)
        os.makedirs(persist_directory, exist_ok=True)
        
        # Prepare document content
        combined_text = f"Question: {question}\n"
        if query:
            combined_text += f"Query: {query}\n"
        if response:
            combined_text += f"Response: {response}"
        if error_message:
            combined_text += f"\nError: {error_message}"

        # Create metadata for vector store
        vector_metadata = {
            "timestamp": str(datetime.now().isoformat()),
            "teacher_id": str(teacher_id),
            "question": str(question),
            "sql_query": str(query) if query else "N/A",
            "has_query": "true" if query else "false",
            "has_response": "true" if response else "false",
            "tag": str(tag) if tag else "none"
        }

        doc = Document(
            page_content=combined_text,
            metadata=vector_metadata
        )

        # Initialize embeddings and vector store
        embeddings = get_embeddings()
        
        if os.path.exists(persist_directory):
            vector_store = Chroma(
                persist_directory=persist_directory,
                embedding_function=embeddings
            )
            vector_store.add_documents([doc])
        else:
            vector_store = Chroma.from_documents(
                documents=[doc],
                persist_directory=persist_directory,
                embedding_function=embeddings
            )
        
        vector_store.persist()
        
    except Exception as e:
        print(f"Error in vector store operations: {str(e)}")

def verify_conversation_storage(teacher_id: str, question: str) -> dict:
    """Verify that a conversation was properly stored."""
    status = {
        "json_stored": False,
        "vector_stored": False,
        "errors": []
    }

    try:
        # Check JSON storage
        from memory.conversation import load_chat_history
        messages = load_chat_history(teacher_id)
        for msg in messages:
            if msg.get("role") == "user" and msg.get("content") == question:
                status["json_stored"] = True
                break
                        
        # Check vector storage
        persist_directory = get_vector_store_path(teacher_id)
        if os.path.exists(persist_directory):
            embeddings = get_embeddings()
            vector_store = Chroma(
                persist_directory=persist_directory,
                embedding_function=embeddings
            )
            
            results = vector_store.similarity_search(question, k=1)
            if results and any(question in doc.page_content for doc in results):
                status["vector_stored"] = True
                
    except Exception as e:
        status["errors"].append(str(e))
        
    return status