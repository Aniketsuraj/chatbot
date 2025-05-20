from langchain.schema import AIMessage, HumanMessage
from datetime import datetime
import json

def format_chat_history(messages: list) -> str:
    """Format chat history into a string for context."""
    if not messages:
        return ""
        
    formatted_history = []
    
    for msg in messages:
        if isinstance(msg, HumanMessage):
            formatted_history.append(f"Human: {msg.content}")
        elif isinstance(msg, AIMessage):
            # Extract only the main content, not follow-up questions
            content = msg.content
            if "Follow-Up Questions:" in content:
                content = content.split("Follow-Up Questions:")[0].strip()
            formatted_history.append(f"Assistant: {content}")
        elif isinstance(msg, dict):
            # Handle message format from loaded history
            role = "Human" if msg.get('role') == 'user' else "Assistant"
            content = msg.get('content', '')
            if role == "Assistant" and "Follow-Up Questions:" in content:
                content = content.split("Follow-Up Questions:")[0].strip()
            formatted_history.append(f"{role}: {content}")
        else:
            # Fallback handling for other message types
            formatted_history.append(str(msg))
    
    # Return most recent messages for context (last 5 exchanges)
    return "\n".join(formatted_history[-5:])

def clean_metadata(metadata: dict) -> dict:
    """
    Clean metadata to ensure only simple types (str, int, float) are included.
    """
    cleaned = {}
    for key, value in metadata.items():
        # Convert None to string
        if value is None:
            cleaned[key] = 'N/A'
        # Convert datetime to string
        elif isinstance(value, datetime):
            cleaned[key] = value.isoformat()
        # Convert boolean to string
        elif isinstance(value, bool):
            cleaned[key] = str(value)
        # Keep strings, integers, and floats
        elif isinstance(value, (str, int, float)):
            cleaned[key] = value
        # Convert everything else to string
        else:
            try:
                cleaned[key] = str(value)
            except:
                cleaned[key] = 'N/A'
    return cleaned

def dataframe_to_json(df):
    """Convert DataFrame to JSON-compatible format safely."""
    if df is None or df.empty:
        return None
    
    try:
        return df.to_dict(orient='records')
    except Exception as e:
        print(f"Error converting DataFrame to JSON: {e}")
        return None

def json_to_dataframe(json_data):
    """Convert JSON data back to DataFrame safely."""
    if json_data is None:
        return None
    
    import pandas as pd
    try:
        return pd.DataFrame(json_data)
    except Exception as e:
        print(f"Error converting JSON to DataFrame: {e}")
        return pd.DataFrame()