import streamlit as st
from langchain.memory import ConversationBufferMemory
from langchain.schema import AIMessage, HumanMessage
import pandas as pd
import json
import os
from datetime import datetime

@st.cache_resource
def init_memory():
    """Initialize and cache conversation memory."""
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    return st.session_state.memory

def load_chat_history(teacher_id: str) -> list:
    """Load chat history from JSON file and convert to message format."""
    filename = f"logs/teacher_{teacher_id}_conversations.json"
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
            messages = []
            for entry in data:
                if entry.get("tag") is None:
                    messages.append({
                        "role": "user",
                        "content": entry["question"]
                    })
                else:
                    messages.append({
                        "role": "assistant",
                        "content": entry["response"],
                        "dataframe": pd.DataFrame(entry["sql_result"]) if entry.get("sql_result") else None,
                        "query": entry.get("sql_query", "N/A"),
                        "result": entry.get("sql_result"),
                        "tag": entry["tag"]
                    })
            return messages
    except (FileNotFoundError, json.JSONDecodeError):
        return []

def clear_json_file(teacher_id: str):
    """Clear the JSON file for a specific teacher."""
    filename = f"logs/teacher_{teacher_id}_conversations.json"
    if os.path.exists(filename):
        with open(filename, 'w') as f:
            json.dump([], f)
        print(f"JSON file cleared for teacher {teacher_id}")

def clear_chat_history(teacher_id: str):
    """Clear all chat history including JSON logs."""    
    # Clear JSON logs
    clear_json_file(teacher_id)

def save_conversation_entry(
    question: str, 
    query: str = None, 
    response: str = None, 
    sql_result: pd.DataFrame = None, 
    tag: str = None,
    teacher_id: str = None,
    error_message: str = None
):
    """Save a single conversation entry to JSON log file."""
    try:
        # Create necessary directories
        os.makedirs('logs', exist_ok=True)
        
        # Ensure teacher_id is available
        if not teacher_id:
            teacher_id = st.session_state.get('teacher_id', 'default')
            
        filename = f"logs/teacher_{teacher_id}_conversations.json"
        
        # Convert SQL results to dict if available
        sql_result_dict = None
        if isinstance(sql_result, pd.DataFrame) and not sql_result.empty:
            try:
                sql_result_dict = sql_result.to_dict(orient='records')
            except Exception as e:
                error_message = f"Error converting DataFrame: {str(e)}"
                print(error_message)

        # Create JSON entry with all fields
        new_entry = {
            "timestamp": datetime.now().isoformat(),
            "teacher_id": str(teacher_id),
            "question": str(question),
            "query": str(query) if query else None,
            "response": str(response) if response else None,
            "sql_result": sql_result_dict,
            "tag": str(tag) if tag else None,
            "error": str(error_message) if error_message else None
        }
        
        try:
            if os.path.exists(filename):
                with open(filename, 'r', encoding='utf-8') as f:
                    try:
                        data = json.load(f)
                    except json.JSONDecodeError:
                        print(f"Error reading JSON file. Creating new file.")
                        data = []
            else:
                data = []
                
        except Exception as e:
            print(f"Error accessing JSON file: {str(e)}")
            data = []
            
        # Append new entry
        data.append(new_entry)
        
        # Save updated JSON file
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving to JSON file: {str(e)}")
            
    except Exception as e:
        error_msg = f"Fatal error in save_conversation_entry: {str(e)}"
        print(error_msg)