import streamlit as st
from config.settings import DEFAULT_TEACHER_ID

def initialize_session_state():
    """Initialize all session state variables."""
    if "teacher_id" not in st.session_state:
        st.session_state.teacher_id = DEFAULT_TEACHER_ID
    if "current_df" not in st.session_state:
        st.session_state.current_df = None
    if "process_steps" not in st.session_state:
        st.session_state.process_steps = {}
    if "expanded" not in st.session_state:
        st.session_state.expanded = {}

def update_teacher_id(new_id, memory):
    """Update the teacher ID and reset associated state."""
    if new_id != st.session_state.teacher_id:
        st.session_state.teacher_id = new_id
        st.session_state.current_df = None
        if memory:
            memory.clear()
        from memory.conversation import clear_chat_history
        clear_chat_history(st.session_state.teacher_id)
        return True
    return False