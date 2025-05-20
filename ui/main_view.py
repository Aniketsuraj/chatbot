import streamlit as st
import pandas as pd
from langchain.schema import AIMessage, HumanMessage

from ai.chains import StepCollector
from ai.llm import StreamlitCallbackHandler
from ai.handlers import process_question

def render_main_view(memory):
    """
    Render the main chat interface.
    
    Args:
        memory: ConversationBufferMemory instance
    """
    # Get conversation history
    messages = memory.chat_memory.messages if memory else []

    # Display messages in the chat UI
    chat_container = st.container()
    with chat_container:
        for i, msg in enumerate(messages):
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            with st.chat_message(role):
                st.markdown(msg.content)

                if role == "assistant":
                    # Get the corresponding user question
                    if i > 0 and isinstance(messages[i-1], HumanMessage):
                        question = messages[i-1].content.lower()
                        
                        # Display dataframe if available
                        if hasattr(msg, 'additional_kwargs'):
                            additional_kwargs = msg.additional_kwargs
                            if 'dataframe' in additional_kwargs and additional_kwargs['dataframe'] is not None:
                                df = additional_kwargs['dataframe']
                                if isinstance(df, pd.DataFrame) and not df.empty:
                                    st.dataframe(df, use_container_width=True)
                        
                        # Add show process button if we have process steps for this question
                        if question in st.session_state.process_steps:
                            button_key = f"process_button_{i}"
                            if st.button("Show Process", key=button_key):
                                st.session_state.expanded[button_key] = not st.session_state.expanded.get(button_key, False)
                            
                            # Show steps if expanded
                            if st.session_state.expanded.get(button_key, False):
                                with st.expander("Process Steps", expanded=True):
                                    steps = st.session_state.process_steps[question]
                                    for step_name, step_value in steps.items():
                                        st.markdown(f"**{step_name}**")
                                        st.text(step_value)
                                        st.markdown("---")

    # Chat input handling
    if question := st.chat_input("Ask me anything..."):
        # Create a new process collector for this question
        process = StepCollector()
        process.add_step("1: teacher_id input", st.session_state.teacher_id)
        process.add_step("2: Question input", question.lower())
        
        with st.chat_message("user"):
            st.markdown(question.lower())
            memory.chat_memory.add_message(HumanMessage(content=question.lower()))

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            df_placeholder = st.empty()

            try:
                with st.spinner('Thinking...'):
                    # Set up streaming handler
                    streaming_handler = StreamlitCallbackHandler(chat_container, message_placeholder)
                    
                    # Process the question
                    response, results, tool_type = process_question(
                        question=question.lower(),
                        streaming_handler=streaming_handler,
                        messages=memory.chat_memory.messages,
                        memory=memory,
                        process_collector=process
                    )
                    
                    # Handle DataFrame display
                    if results is not None and "dataframe" in results:
                        df = results.get("dataframe", pd.DataFrame())
                        if isinstance(df, pd.DataFrame) and not df.empty:
                            with df_placeholder:
                                st.dataframe(df, use_container_width=True)

                            # Add response with DataFrame to memory
                            memory.chat_memory.add_message(
                                AIMessage(
                                    content=response,
                                    additional_kwargs={"dataframe": df}
                                )
                            )
                        else:
                            memory.chat_memory.add_message(AIMessage(content=response))
                    else:
                        memory.chat_memory.add_message(AIMessage(content=response))
                    
                    # Store process steps in session state
                    st.session_state.process_steps[question.lower()] = process.get_steps()

                    st.rerun()

            except Exception as e:
                error_message = f"An error occurred: {str(e)}"
                st.error(error_message)
                process.add_step("Error", error_message)
                st.session_state.process_steps[question.lower()] = process.get_steps()
                memory.chat_memory.add_message(AIMessage(content=error_message))