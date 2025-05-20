import streamlit as st
from memory.conversation import clear_chat_history
from utils.session import update_teacher_id

def render_sidebar(memory):
    """
    Render the sidebar with settings.
    
    Args:
        memory: ConversationBufferMemory instance
    """
    with st.sidebar:
        st.header("Settings")
        
        # Teacher ID input
        new_teacher_id = st.text_input("Teacher ID", value=st.session_state.teacher_id)
        
        # Update teacher ID if changed
        if update_teacher_id(new_teacher_id, memory):
            st.rerun()

        # Clear chat history button
        if st.button("Clear Chat History"):
            memory.clear()
            st.session_state.current_df = None
            clear_chat_history(st.session_state.teacher_id)
            st.rerun()
            
        # Add additional settings as needed
        st.divider()
        st.markdown("### About")
        st.markdown("""
        This AI Assistant helps teachers analyze student performance data.
        Ask questions about classes, students, and learning outcomes to get insights.
        """)