import streamlit as st
from langchain_community.utilities import SQLDatabase
from config.settings import DB_URI

@st.cache_resource
def init_db():
    """Initialize and cache SQLDatabase connection."""
    return SQLDatabase.from_uri(DB_URI)