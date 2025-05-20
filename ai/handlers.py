import streamlit as st
import pandas as pd
from langchain.schema import AIMessage

from ai.chains import create_chain, StepCollector
from ai.llm import get_chat_llm
from ai.prompts import get_prompts
from database.connection import init_db
from memory.conversation import save_conversation_entry
from memory.vector_store import save_to_vector_store
from utils.formatters import format_chat_history

def handle_query_data(question, streaming_handler, messages, tag, memory, process_collector=None):
    """
    Handle data queries with SQL chain with process tracking.
    
    Args:
        question: User question string
        streaming_handler: Callback handler for streaming output
        messages: List of conversation messages
        tag: Question type tag
        memory: ConversationBufferMemory instance
        process_collector: Optional StepCollector instance for debugging
        
    Returns:
        tuple: (response, results)
    """
    # Initialize LLM with streaming
    llm = get_chat_llm(streaming_handler)
    
    # Format chat history properly for messages
    chat_history = format_chat_history(messages)
    
    # Initialize database
    db = init_db()
    
    # Create and invoke the chain
    chain = create_chain(db, st.session_state.teacher_id, messages, memory, process_collector)
   
    results = chain.invoke({
        "question": question.lower(),
        "teacher_id": st.session_state.teacher_id,
        "chat_history": chat_history
    })
    
    # Store current dataframe in session state
    st.session_state.current_df = results["dataframe"]
    
    # Get the answer prompt template
    _, answer_prompt, _, _ = get_prompts()

    # Generate response using the answer prompt template
    response = llm.invoke(
        answer_prompt.format(
            chat_history=chat_history,
            question=question.lower(),
            query=results["query"],
            result=results["result"]
        )
    )
    
    if process_collector:
        process_collector.add_step("7: the answer", response)

    # Save conversation data
    save_conversation_entry(
        question=question.lower(), 
        query=results["query"], 
        response=response,
        sql_result=results["dataframe"],
        tag=tag,
        teacher_id=st.session_state.teacher_id
    )
    
    save_to_vector_store(
        question=question.lower(), 
        query=results["query"], 
        response=response,
        sql_result=results["dataframe"],
        tag=tag,
        teacher_id=st.session_state.teacher_id
    )

    return response, results

def handle_general_chat(question, streaming_handler, messages, tag, process_collector=None):
    """
    Handle general chat questions.
    
    Args:
        question: User question string
        streaming_handler: Callback handler for streaming output
        messages: List of conversation messages
        tag: Question type tag
        process_collector: Optional StepCollector instance for debugging
        
    Returns:
        str: Response text
    """
    # Initialize LLM with streaming
    llm = get_chat_llm(streaming_handler)

    # Format chat history
    chat_history = format_chat_history(messages)

    # Get the general chat prompt
    _, _, _, general_chat_prompt = get_prompts()

    # Generate response
    response = llm.invoke(general_chat_prompt.format(
        chat_history=chat_history, 
        question=question
    ))
    
    if process_collector:
        process_collector.add_step("4: the general answer", response)
    
    # Save conversation data
    save_conversation_entry(
        question=question,
        response=response,
        tag=tag,
        teacher_id=st.session_state.teacher_id
    )
    
    save_to_vector_store(
        question=question,
        response=response,
        tag=tag,
        teacher_id=st.session_state.teacher_id
    )
    
    return response

def generate_follow_up_questions(question, streaming_handler, messages, response, results, memory, process_collector=None):
    """
    Generate follow-up questions based on query results.
    
    Args:
        question: Original user question
        streaming_handler: Callback handler for streaming output
        messages: List of conversation messages
        response: Response text from the main query
        results: Query results dictionary
        memory: ConversationBufferMemory instance
        process_collector: Optional StepCollector instance for debugging
        
    Returns:
        str: Follow-up questions text
    """
    # Initialize LLM
    llm = get_chat_llm(streaming_handler)

    # Format chat history
    chat_context = format_chat_history(messages)
    
    # Get database schema information
    db = init_db()
    table_info = db.get_table_info()

    # Get the follow-up questions prompt
    _, _, follow_up_prompt, _ = get_prompts()

    # Extract result data for the prompt
    result_data = ""
    if isinstance(results, dict) and "dataframe" in results:
        df = results.get("dataframe")
        if isinstance(df, pd.DataFrame) and not df.empty:
            result_data = df.to_string(index=False)
    
    # Prepare the input
    follow_up_input = {
        "input": question,
        "response": response,
        "result": result_data,
        "chat_history": chat_context,
        "table_info": table_info
    }

    # Generate follow-up questions
    follow_up_response = llm.invoke(follow_up_prompt.format(**follow_up_input))
    
    if process_collector:
        process_collector.add_step("8: the follow up questions", follow_up_response)

    # Save to conversation history
    save_conversation_entry(
        question=question,
        response=f"Follow-Up Questions:\n{follow_up_response}",
        tag="follow_up_questions",
        teacher_id=st.session_state.teacher_id
    )
    
    save_to_vector_store(
        question=question,
        response=f"Follow-Up Questions:\n{follow_up_response}",
        tag="follow_up_questions",
        teacher_id=st.session_state.teacher_id
    )

    return follow_up_response

def process_question(question, streaming_handler, messages, memory, process_collector=None):
    """
    Process a user question by determining its type and handling appropriately.
    
    Args:
        question: User question string
        streaming_handler: Callback handler for streaming output
        messages: List of conversation messages
        memory: ConversationBufferMemory instance
        process_collector: Optional StepCollector instance for debugging
        
    Returns:
        tuple: (response, results, tool_type)
    """
    from ai.tagger import create_tagger
    from ai.llm import get_llm
    
    try:
        # Format chat history
        chat_history = format_chat_history(messages)
        
        # Create a tagger and classify the question
        tagger_func = create_tagger(get_llm(temperature=0.1))
        tool_type = tagger_func({
            "question": question.lower(),
            "chat_history": chat_history
        })
        
        if process_collector:
            process_collector.add_step("3: tagger", tool_type)

        if tool_type == "query_data":
            # Process query data with chat history
            response, results = handle_query_data(
                question, 
                streaming_handler, 
                messages,  
                tag=tool_type, 
                memory=memory,
                process_collector=process_collector
            )

            # Generate follow-up questions
            follow_up_questions = generate_follow_up_questions(
                question=question.lower(),
                streaming_handler=streaming_handler,
                response=response,
                results=results,
                messages=messages,
                memory=memory,
                process_collector=process_collector
            )

            # Combine main response and follow-up questions
            full_response = f"{response}\n\nFollow-Up Questions:\n{follow_up_questions}"
            
            return full_response, results, tool_type
        else:
            # Handle general chat
            response = handle_general_chat(
                question.lower(), 
                streaming_handler, 
                messages, 
                tag=tool_type,
                process_collector=process_collector
            )
            
            return response, None, tool_type
            
    except Exception as e:
        error_message = f"An error occurred while processing your question: {str(e)}"
        if process_collector:
            process_collector.add_step("Error", error_message)
            
        print(error_message)
        save_conversation_entry(
            question=question.lower(),
            response=error_message,
            error_message=str(e),
            teacher_id=st.session_state.teacher_id
        )
        
        return error_message, None, "error"