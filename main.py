import streamlit as st
from utils.session import initialize_session_state
from memory.conversation import init_memory
from ui.main_view import render_main_view
from ui.sidebar import render_sidebar

# App configuration
st.set_page_config(
    page_title="AI Assistant",
    page_icon="🤖",
    layout="wide"
)

def main():
    """Main application function."""
    st.title("AI Assistant")

    # Initialize session state and memory
    initialize_session_state()
    memory = init_memory()
    
    # Initialize steps dictionary if not exists
    if "process_steps" not in st.session_state:
        st.session_state.process_steps = {}
    
    # Initialize expanded state for process buttons
    if "expanded" not in st.session_state:
        st.session_state.expanded = {}

    # Render sidebar
    render_sidebar(memory)
    
    # Render main view
    render_main_view(memory)

if __name__ == "__main__":
    main()