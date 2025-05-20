from langchain_core.runnables import RunnablePassthrough
from langchain.chains import create_sql_query_chain
from langchain.schema import AIMessage, HumanMessage
import pandas as pd

from ai.prompts import get_prompts
from database.queries import clean_query, DataFrameSQLTool
from memory.vector_store import get_similar_conversations
from utils.formatters import format_chat_history

class StepCollector:
    """Utility class for collecting processing steps for debugging."""
    def __init__(self):
        self.steps = {}
    
    def add_step(self, step_name, step_output):
        self.steps[step_name] = step_output
    
    def get_steps(self):
        return self.steps

def create_chain(db, teacher_id, messages, memory, process_collector=None):
    """Create a chain that uses vector store for finding similar SQL queries with process tracking.
    
    Args:
        db: SQLDatabase instance
        teacher_id: Teacher ID string
        messages: List of conversation messages
        memory: ConversationBufferMemory instance
        process_collector: Optional StepCollector instance for debugging
        
    Returns:
        A LangChain runnable chain
    """
    sql_prompt, answer_prompt, _, _ = get_prompts()
    execute_query = DataFrameSQLTool(db=db)
    
    def write_query(inputs: dict) -> str:
        """Generate SQL query from natural language input."""
        try:
            # Format chat history properly
            if memory and memory.chat_memory.messages:
                chat_history = format_chat_history(memory.chat_memory.messages)
            else:
                chat_history = ""
            
            similar_queries = get_similar_conversations(inputs["question"], teacher_id)
            
            similar_examples = []
            for query in similar_queries:
                if query != "N/A":
                    similar_examples.append({
                        "input": inputs["question"],
                        "query": query
                    })
            
            all_examples = sql_prompt.examples + similar_examples
            
            # Create the enhanced prompt
            from langchain_core.prompts import FewShotPromptTemplate
            enhanced_prompt = FewShotPromptTemplate(
                examples=all_examples,
                example_prompt=sql_prompt.example_prompt,
                prefix=sql_prompt.prefix,
                suffix=sql_prompt.suffix,
                input_variables=sql_prompt.input_variables
            )
            
            # Create and execute the SQL chain
            from ai.llm import get_query_llm
            chain = create_sql_query_chain(
                llm=get_query_llm(),
                db=db,
                prompt=enhanced_prompt
            )
            
            # Format the query inputs
            query_inputs = {
                "question": inputs["question"],
                "teacher_id": teacher_id,
                "chat_history": chat_history,
                "table_info": db.get_table_info()
            }
            
            # Generate the query
            query = chain.invoke(query_inputs)
            
            # Clean and format the query
            query = clean_query(query)
            
            if process_collector:
                process_collector.add_step("5: generated and cleaned sql", query)

            print(query)
            
            return query
            
        except Exception as e:
            error_msg = f"Error generating SQL query: {e}"
            if process_collector:
                process_collector.add_step("5: generated and cleaned sql", error_msg)
            print(error_msg)
            return ""

    def process_query(inputs: dict) -> dict:
        """Process SQL query and return results."""
        try:
            if not inputs.get("query"):
                if process_collector:
                    process_collector.add_step("6: sql result", "No query was generated")
                    
                return {
                    "question": inputs["question"],
                    "query": "",
                    "result": "I couldn't generate a proper query. Please rephrase your question.",
                    "dataframe": pd.DataFrame()
                }
                
            query = inputs["query"]
            result_df = execute_query.run(query.lower())
            
            if result_df.empty:
                if process_collector:
                    process_collector.add_step("6: sql result", "Query executed but returned no data")
                    
                return {
                    "question": inputs["question"],
                    "query": query,
                    "result": "No data found for your query. Please try a different question.",
                    "dataframe": pd.DataFrame()
                }
            
            if process_collector:
                process_collector.add_step("6: sql result", result_df.to_string(index=False))
                
            return {
                "question": inputs["question"],
                "query": query,
                "result": result_df.to_string(index=False),
                "dataframe": result_df
            }
            
        except Exception as e:
            error_msg = f"Error executing query: {e}"
            if process_collector:
                process_collector.add_step("6: sql result", error_msg)
                
            print(error_msg)
            return {
                "question": inputs["question"],
                "query": "",
                "result": "An error occurred while processing your request. Please try again.",
                "dataframe": pd.DataFrame()
            }
    
    # Create the chain
    chain = (
        RunnablePassthrough.assign(query=write_query)
        | process_query
    )

    return chain