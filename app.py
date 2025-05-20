import streamlit as st
from langchain_community.utilities import SQLDatabase
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.chains import create_sql_query_chain
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from typing import Dict
import pandas as pd
from datetime import datetime
import re
import json
import os
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.schema import AIMessage, HumanMessage
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

st.set_page_config(
    page_title="AI Assistant",
    page_icon="🤖",
    layout="wide"
)

class StepCollector:
    def __init__(self):
        self.steps = {}
    
    def add_step(self, step_name, step_output):
        self.steps[step_name] = step_output
    
    def get_steps(self):
        return self.steps

def initialize_session_state():
    """Initialize all session state variables"""
    if "teacher_id" not in st.session_state:
        st.session_state.teacher_id = "8933"
    if "current_df" not in st.session_state:
        st.session_state.current_df = None

@st.cache_resource
def init_memory():
    """Initialize and cache conversation memory"""
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    return st.session_state.memory

def load_chat_history(teacher_id: str) -> list:
    """Load chat history from JSON file and convert to message format"""
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

def format_chat_history(messages: list) -> str:
    """Format chat history into a string for context"""
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

# def clear_vector_store(teacher_id: str):
#     """Clear the vector store directory for a specific teacher."""
#     vector_store_path = f'vector_store/teacher_{teacher_id}'

#     if not os.path.exists(vector_store_path):
#         print(f"No vector store found for teacher {teacher_id}.")
#         return

#     print(f"Attempting to clear vector store at: {vector_store_path}")
    
#     try:
#         # Step 1: Detect and terminate processes holding the directory
#         for proc in psutil.process_iter(['pid', 'name', 'open_files']):
#             try:
#                 for file in proc.open_files():
#                     if vector_store_path in file.path:
#                         print(f"Terminating process: {proc.name()} (PID: {proc.pid})")
#                         proc.terminate()  # Terminate the process
#                         proc.wait(timeout=5)  # Wait for process to exit
#             except (psutil.AccessDenied, psutil.NoSuchProcess):
#                 continue

#         # Step 2: Retry deleting the directory
#         for attempt in range(5):
#             try:
#                 shutil.rmtree(vector_store_path)
#                 print(f"Successfully cleared vector store for teacher {teacher_id}")
#                 return
#             except Exception as e:
#                 print(f"Attempt {attempt + 1} failed: {e}")
#                 time.sleep(2)  # Wait before retrying

#         raise RuntimeError(f"Failed to clear vector store after multiple attempts: {vector_store_path}")

#     except Exception as e:
#         print(f"Error clearing vector store for teacher {teacher_id}: {e}")



def clear_json_file(teacher_id: str):
    """Clear the JSON file for a specific teacher"""
    filename = f"logs/teacher_{teacher_id}_conversations.json"
    if os.path.exists(filename):
        with open(filename, 'w') as f:
            json.dump([], f)
        print(f"JSON file cleared for teacher {teacher_id}")

def clear_chat_history(teacher_id: str):
    """Clear all chat history including vector store and JSON logs"""
    # Clear vector store
    # clear_vector_store(teacher_id)
    
    # Clear JSON logs
    clear_json_file(teacher_id)

# class QuestionTag(BaseModel):
#     """Tag for classifying the question type"""
#     tool: str = Field(
#         description="The tool to use. Must be either 'query_data' or 'general_chat'",
#         enum=["query_data", "general_chat"]
#     )
#     reason: str = Field(description="Brief reason for selecting this tool")

class StreamlitCallbackHandler(StreamingStdOutCallbackHandler):
    def __init__(self, container, text_placeholder):
        self.container = container
        self.text_placeholder = text_placeholder
        self.text = ""

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.text_placeholder.markdown(self.text)

class DataFrameSQLTool:
    def __init__(self, db: SQLDatabase):
        self.db = db
    
    def clean_query(self, query: str) -> str:
        """Clean and validate SQL query"""
        try:
            # Remove extra parentheses
            query = query.strip()
            while query.count('(') != query.count(')'):
                if query.count('(') > query.count(')'):
                    query = query.rstrip(')')
                else:
                    query = query.lstrip('(')
                    
            # Ensure query ends with semicolon
            if not query.endswith(';'):
                query += ';'
                
            # Clean up whitespace and line breaks
            query = ' '.join(query.split())
            
            return query
            
        except Exception as e:
            print(f"Error cleaning query: {e}")
            return query
    
    def run(self, query: str) -> pd.DataFrame:
        """Execute SQL query and return results"""
        try:
            # Clean the query first
            cleaned_query = self.clean_query(query)
            print(f"Executing query: {cleaned_query}")
            
            with self.db._engine.connect() as conn:
                try:
                    # Execute the cleaned query
                    result = pd.read_sql_query(cleaned_query, conn)
                    return result if not result.empty else pd.DataFrame()
                        
                except Exception as e:
                    print(f"Error executing query: {e}")
                    return pd.DataFrame()
                    
        except Exception as e:
            print(f"Error in query execution: {e}")
            return pd.DataFrame()
@st.cache_resource
def init_db():
    return SQLDatabase.from_uri("sqlite:///acadally.db")

@st.cache_data
def get_prompts():
    example_prompt = PromptTemplate(
        input_variables=["input", "query"],
        template="Input: {input}\nSQL Query: {query}"
    )

    examples = [
        {
            "input": "Give me the overall performance summary of all the classes",
            "query": """
                    SELECT DISTINCT s.section_name, s.section_CR, s.section_LI 
                    FROM section s 
                    WHERE s.teacher_id = {teacher_id}
                    ORDER BY s.section_CR DESC;
                    """
        },
        {
            "input": "How do I improve the performance of class 7a?",
            "query": """
                    SELECT DISTINCT st.student_name, st.student_CR, st.student_LI, 
                        st.chapter_name, st.chapter_CR, st.chapter_LI
                    FROM student st 
                    WHERE st.teacher_id = {teacher_id} AND st.section_name = 'class 7a'
                    ORDER BY st.student_CR ASC;
                    """
        },
        {
            "input": "How many learning units attempted by class 6a?",
            "query": """
                    SELECT DISTINCT s.section_name, s.section_level_engaged_LU_count as total_attempted_units 
                    FROM section s  
                    WHERE s.section_name = 'class 6a' AND s.teacher_id = {teacher_id};
                    """
        },     
        {
            "input": "List of students engaged in chapter friction",
            "query": """
                    SELECT DISTINCT st.student_name, st.section_name  
                    FROM student st
                    WHERE st.teacher_id = {teacher_id} AND st.chapter_name = 'friction' 
                    AND st.chapter_level_engaged_LU_count != -1;
                    """
        },
        {
            "input": "What is the total number of learning units assigned to each student in class 6a?",
            "query": """
                    SELECT DISTINCT st.student_name, st.student_level_total_LU_count as total_assigned_units 
                    FROM student st 
                    WHERE st.section_name = 'class 6a' AND st.teacher_id = {teacher_id};
                    """
        },
        {
            "input": "Which class has the highest overall performance, and how many learning units have been engaged in class 8d?",
            "query": """
                    SELECT DISTINCT s.section_name, s.section_CR, s.section_LI, s.section_level_engaged_LU_count 
                    FROM section s
                    WHERE s.teacher_id = {teacher_id} AND s.section_name = 'class 8d'
                    ORDER BY s.section_CR DESC;
                    """
        },
        {
            "input": "What's the average time spent on each chapter?",
            "query": """
                    SELECT s.chapter_name, s.section_name,
                    ROUND(AVG(JULIANDAY(s.chapter_end_timestamp) - JULIANDAY(s.chapter_start_timestamp)), 1) as avg_days,
                    AVG(s.chapter_CR) as avg_completion_rate,
                    AVG(s.chapter_LI) as avg_learning_index
                    FROM section s
                    WHERE s.teacher_id = {teacher_id} AND s.chapter_end_timestamp IS NOT NULL
                    GROUP BY s.chapter_name, s.section_name
                    ORDER BY avg_days DESC;
                    """
        },
        {
            "input": "Give me my improvement areas",
            "query": """
                    SELECT DISTINCT st.chapter_name, AVG(st.chapter_CR) as avg_cr, AVG(st.chapter_LI) as avg_li, st.section_name
                    FROM student st
                    WHERE st.teacher_id = {teacher_id} AND st.chapter_CR < 50
                    GROUP BY st.section_name, st.chapter_name
                    ORDER BY avg_cr ASC
                    LIMIT 5;
                    """
        },
        {
            "input": "How many learning units have been engaged across all sections?",
            "query": """
                    SELECT DISTINCT s.section_name, s.section_level_engaged_LU_count
                    FROM section s 
                    WHERE s.teacher_id = {teacher_id} AND s.section_level_engaged_LU_count != -1
                    ORDER BY s.section_level_engaged_LU_count DESC;
                    """
        },
        {
            "input": "Give me my worst performing classes",
            "query": """
                    SELECT DISTINCT s.section_name, s.section_CR, s.section_LI
                    FROM section s
                    WHERE s.teacher_id = {teacher_id}
                    ORDER BY s.section_CR ASC
                    LIMIT 5;
                    """
        },
        {
            "input": "Which chapters have the best performance across all classes?",
            "query": """
                    SELECT s.chapter_name, AVG(s.chapter_CR) as avg_cr, AVG(s.chapter_LI) as avg_li,
                    COUNT(DISTINCT s.section_name) as num_sections
                    FROM section s
                    WHERE s.teacher_id = {teacher_id} AND s.chapter_end_timestamp IS NOT NULL
                    GROUP BY s.chapter_name
                    HAVING avg_cr >= 60
                    ORDER BY avg_cr DESC
                    LIMIT 5;
                    """
        },
        {
            "input": "Top performing students in class 6D",
            "query": """
                    SELECT DISTINCT st.student_name, st.student_CR, st.student_LI 
                    FROM student st 
                    WHERE st.section_name = 'class 6d' AND st.teacher_id = {teacher_id} 
                    ORDER BY st.student_CR DESC, st.student_LI DESC 
                    LIMIT 5;
                    """
        },
        {
            "input": "Recent performance of class 8e",
            "query": """
                    SELECT DISTINCT
                    s.chapter_name,
                    s.section_name,
                    s.chapter_CR as completion_rate,
                    s.chapter_LI as learning_index,
                    s.chapter_level_engaged_LU_count as attempted_units,
                    s.chapter_level_LG_count as learning_gaps,
                    date(s.chapter_start_timestamp) as start_date
                    FROM section s
                    WHERE s.teacher_id = {teacher_id} 
                    AND s.section_name = 'class 8e'
                    AND date(s.chapter_start_timestamp) >= date('now', '-30 days')
                    ORDER BY s.chapter_start_timestamp DESC;
                    """
        },
        {
            "input": "What is the average completion rate among students for the chapters \"Crop Production and Management\" and \"Reproduction in Animals\"?",
            "query": """
                    SELECT st.chapter_name, AVG(st.chapter_CR) AS avg_completion_rate
                    FROM student st
                    WHERE st.teacher_id = {teacher_id} 
                    AND st.chapter_name IN ('Crop Production and Management', 'Reproduction in Animals') 
                    GROUP BY st.chapter_name;
                    """
        },
        {
            "input": "How many chapters are ongoing in each class?",
            "query": """
                    SELECT DISTINCT s.section_name, COUNT(DISTINCT s.chapter_name) as ongoing_chapters 
                    FROM section s 
                    WHERE s.teacher_id = {teacher_id} AND s.ongoing_flag = true 
                    GROUP BY s.section_name
                    ORDER BY ongoing_chapters DESC;
                    """
        }
    ]

    sql_prompt = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix="""You are an SQL expert generating precise queries for an education analytics database. Your task is to translate natural language questions into SQL queries that exactly match the pattern and style of the examples.

## Available Tables
- section: Class-level aggregated data (use for class/section-level analysis)
- student: Individual student performance data (use for student-level analysis)

## Previous Conversation Context
{chat_history}

## CRITICAL: FIELD USAGE RESTRICTION
- ONLY use fields that are explicitly mentioned in the table schema provided in the "Table Schema Details" section
- DO NOT create, infer, or assume any fields that don't exist in the table schema
- If a needed field doesn't exist in the schema, use the closest available field instead
- Never reference columns that aren't explicitly listed in the schema

## CRITICAL: REFERENCE RESOLUTION RULES
1. When a question uses terms like "this class" or "the class", refer to the most recently mentioned class in the conversation
2. When "top students" are mentioned, always match the class from the previous conversation context
3. If a question uses demonstrative pronouns (this, that, these, those), resolve them to the specific entities discussed previously
4. Pay attention to class names (e.g., "class 8d") mentioned in the conversation and maintain consistency
5. If the current question refers to "performance" without specifying a class, use the class from the most recent conversation

## Schema and Performance Metrics:
### Completion Rate (CR) - PRIMARY Performance Measurement
- student_CR: Student's overall completion/performance percentage
- section_CR: Section's overall completion/performance percentage
- chapter_CR: Chapter-specific completion/performance percentage
- topic_CR: Topic-specific completion/performance percentage

### Learning Index (LI) - Secondary Metric
- student_LI: Student's overall learning performance across all chapters
- section_LI: Section's overall learning performance across all chapters
- chapter_LI: Learning performance specific to a chapter
- topic_LI: Learning performance specific to a topic

### Learning Unit (LU) Count Fields
Student-level counts (in student table):
- student_level_LG_count: Learning gaps count (-1 = not attempted)
- student_level_engaged_LU_count: Units attempted by student (-1 = not attempted)
- student_level_total_LU_count: Units assigned to student (-1 = not assigned)

Section-level counts (in section table):
- section_level_LG_count: Learning gaps count (-1 = not attempted)
- section_level_engaged_LU_count: Units attempted by section (-1 = not attempted)
- section_level_total_LU_count: Units assigned to section (-1 = not assigned)

Chapter-level counts:
- chapter_level_LG_count: Learning gaps count in a chapter (-1 = not attempted)
- chapter_level_engaged_LU_count: Units attempted in a chapter (-1 = not attempted)
- chapter_level_total_LU_count: Units assigned in a chapter (-1 = not assigned)

CRQ-level counts:
- CRQ_level_LG_count: Learning gaps in Chapter Readiness Quiz (-1 = not attempted)
- CRQ_level_engaged_LU_count: Units attempted in CRQ (-1 = not attempted)
- CRQ_level_total_LU_count: Units assigned in CRQ (-1 = not assigned)

### Timeline Fields
- chapter_start_timestamp: Chapter start time
- chapter_end_timestamp: Chapter end time (-1 if ongoing)
- ongoing_flag: TRUE =   ongoing, FALSE = completed

## Table Schema Details
{table_info}


## Critical SQL Rules
- ALWAYS Generate or create SQL queries with columns given in Table Schema and Performance Metrics.
- D0 NOT generate SQL queries with table or columns name that are not given in the table schema and performance metrics.
- ALWAYS include 'teacher_id = {teacher_id}' in the WHERE clause
- Use DISTINCT to avoid duplicates
- USE subject from student table for subject name and student_name from student table for student name
- TOP PERFORMANCE IS MEASURED BY CR (Completion Rate) - use ORDER BY with CR DESC for high performers
- LOW PERFORMANCE IS MEASURED BY CR (Completion Rate) - use ORDER BY with CR ASC for improvement areas
- DO NOT use SUM in SQL queries - counts are pre-calculated in the database
- Always use student table (st) for student data and section table (s) for section/class data
- Always access student_name from student table, not section table
- For questions about "engaged" or "attempted" units, use _engaged_LU_count fields
- For checking ongoing chapters, use ongoing_flag = true
- Follow the exact style and format of the example queries
- "Recent" refers to data within the last 30 days
- Do not use JOIN unless absolutely necessary
- Always include appropriate ORDER BY clauses for ranking queries



{top_k}

GENERATE ONLY THE SQL QUERY. NO EXPLANATIONS OR COMMENTS.
MATCH THE STYLE AND APPROACH OF THE EXAMPLE QUERIES EXACTLY.
""",
        suffix="Question: {input}\n\nSQL Query:",
        input_variables=["input", "table_info", "top_k", "teacher_id", "chat_history"]
    )

    answer_prompt = PromptTemplate.from_template(   
        """
You are a helpful assistant for teachers. Your job is to explain student data in simple, everyday language that any teacher can understand without technical knowledge.

## Context
Question: {question}
SQL Query: {query}
SQL Result: {result}
Previous Conversation: {chat_history}

## Guidelines for Your Answer
- Focus on insights, not SQL explanation
- Interpret percentages meaningfully (60%+ is good, below 50% needs improvement)
- Be specific and use exact names from results
- Give large data in form of lists if the sql result is big.
- Do not create answer with new data or Previous Conversation when SQL result is empty . Answer like this as "I do not have data for this question.Try somthing else"
- Structure your answer in paragraphs, not bullet points
- For large result sets, focus on top 2 and bottom 2 examples
- Include 1-2 specific, actionable recommendations for the teacher to improve performance
- NEVER start with phrases like "Based on the results..." or "According to the data..."
- Use exact student/class names from the results
- Keep your answer concise and practical

## Classroom Recommendation Guide:
- Always base recommendations directly on specific data points from the SQL results
- Reference exact scores, names, or metrics when making suggestions (e.g., "Since Jayden scored 42% on fractions...")
- Identify specific content areas with the lowest scores/progress from the SQL results
- Connect recommendations to specific performance patterns visible in the data
- Use comparative data when available (e.g., "While the class averages 75% on vocabulary, reading comprehension is only at 58%")

## IMPORTANT:
- No statistics language or percentiles
- No technical explanations of the data
- No vague general advice - be specific about what to do in the classroom
- Always connect recommendations directly to the specific students/classes in the results
- Never use bullet points - write in conversational paragraphs

## Simple Language Guide:
- ALWAYS say "progress" instead of "completion rate" or "CR"
- ALWAYS say "performance" instead of "learning index" or "LI"
- Say "students who need help" instead of "low performers"
- Say "topics covered" instead of "learning units"
- Say "identified gaps" instead of "learning gaps"
- Say "class" instead of "section"
- Say "started working on" instead of "engaged with"

## Interpretation Benchmarks
- Performance (LI) above 60%: Good
- Performance (LI) 50-60%: Satisfactory 
- Performance (LI) below 50%: Needs improvement
- Progress (CR) above 60%: Good
- Progress (CR) 50-60%: Satisfactory
- Progress (CR) below 50%: Needs improvement

## Output Format
## Output Format
Answer: [Direct answer to the question with specific insights from the data]
Recommendation : [1-2 specific content areas or skills revealed by the data that need attention]
        """
    )

    return sql_prompt, answer_prompt 

def clean_query(query: str) -> str:
    query = re.sub(r'```sql\s*', '', query)
    query = re.sub(r'\s*```', '', query)
    query = re.sub(r'\s+', ' ', query)
    if "SELECT" in query:
        query = query.split("SELECT")[-1]
        query = query.split(";")[0]
        query = "SELECT " + query

    # Remove parentheses from the start and end of the query
    query = query.strip()
    while (query.startswith("(") and query.endswith(")")) or query.startswith("((") or query.endswith("))"):
        query = query.strip("()")

    # Remove any unmatched closing parentheses
    if query.count("(") < query.count(")"):
        query = query[:query.rfind(")")]
    
    return query.strip()



def create_chain(db, teacher_id: str, messages: list, memory, process_collector=None):
    """Create a chain that uses vector store for finding similar SQL queries with process tracking"""
    sql_prompt, answer_prompt = get_prompts()
    execute_query = DataFrameSQLTool(db=db)
    
    llm1 = Ollama(
        model="gemma3:12b",      
        temperature=0.3,
    )
    
    def get_similar_conversations(question: str) -> list:
        try:
            persist_directory = f'vector_store/teacher_{teacher_id}'
            embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
            
            if not os.path.exists(persist_directory):
                if process_collector:
                    process_collector.add_step("4: similar questions retrieval", "No vector store found for this teacher.")
                return []
                
            vector_store = Chroma(
                persist_directory=persist_directory,
                embedding_function=embeddings
            )
            
            similar_docs = vector_store.similarity_search(
                query=question,
                k=1
            )
            
            similar_queries = [doc.metadata.get("sql_query", "N/A") for doc in similar_docs]
            
            if process_collector:
                if similar_queries and similar_queries[0] != "N/A":
                    process_collector.add_step("4: similar questions retrieval", str(similar_queries[0]))
                else:
                    process_collector.add_step("4: similar questions retrieval", "No similar questions found")
                    
            return similar_queries
            
        except Exception as e:
            if process_collector:
                process_collector.add_step("4: similar questions retrieval", f"Error retrieving similar conversations: {e}")
            print(f"Error retrieving similar conversations: {e}")
            return []
    
    def write_query(inputs: dict) -> str:
        try:
            # Format chat history properly
            if memory and memory.chat_memory.messages:
                chat_history = format_chat_history(memory.chat_memory.messages)
            else:
                chat_history = ""
            
            similar_queries = get_similar_conversations(inputs["question"])
            
            similar_examples = []
            for query in similar_queries:
                if query != "N/A":
                    similar_examples.append({
                        "input": inputs["question"],
                        "query": query
                    })
            
            all_examples = sql_prompt.examples + similar_examples
            
            # Create the enhanced prompt
            enhanced_prompt = FewShotPromptTemplate(
                examples=all_examples,
                example_prompt=sql_prompt.example_prompt,
                prefix=sql_prompt.prefix,
                suffix=sql_prompt.suffix,
                input_variables=sql_prompt.input_variables
            )
            
            # Create and execute the SQL chain
            chain = create_sql_query_chain(
                llm=llm1,
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
    
    chain = (
        RunnablePassthrough.assign(query=write_query)
        | process_query
    )

    return chain


def clean_metadata(metadata: dict) -> dict:
    """
    Clean metadata to ensure only simple types (str, int, float) are included
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

def save_conversation_to_vector_store(
    question: str, 
    query: str = None, 
    response: str = None, 
    sql_result: pd.DataFrame = None, 
    tag: str = None,
    teacher_id: str = None,
    error_message: str = None
):

    """
    Save conversation with improved JSON logging
    """
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
            "query": str(query) if query else None,  # Changed from sql_query to query
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

        # Vector store operations
        try:
            persist_directory = f'vector_store/teacher_{teacher_id}'
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
                "has_query": "true" if query else "false",
                "has_response": "true" if response else "false",
                "tag": str(tag) if tag else "none"
            }

            doc = Document(
                page_content=combined_text,
                metadata=vector_metadata
            )

            # Initialize embeddings and vector store
            embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
            
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
            error_msg = f"Error in vector store operations: {str(e)}"
            print(error_msg)
            # Save error to JSON
            save_conversation_to_vector_store(
                question=question,
                error_message=error_msg,
                teacher_id=teacher_id
            )
            
    except Exception as e:
        error_msg = f"Fatal error in save_conversation_to_vector_store: {str(e)}"
        print(error_msg)



def verify_conversation_storage(teacher_id: str, question: str) -> dict:
    status = {
        "json_stored": False,
        "vector_stored": False,
        "errors": []
    }

    try:
        # Check JSON storage
        filename = f"logs/teacher_{teacher_id}_conversations.json"
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                data = json.load(f)
                for entry in data:
                    if entry["question"] == question:
                        status["json_stored"] = True
                        break
                        
        # Check vector storage
        persist_directory = f'vector_store/teacher_{teacher_id}'
        if os.path.exists(persist_directory):
            embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
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
def create_tagger(llm):
    """Create a simplified chain for tagging questions"""
    template = """
    You are an AI assistant that accurately classifies user questions into two categories.

    Previous conversation context:
    {chat_history}

    CLASSIFICATION RULES:
    
    1. Tag as 'query_data' when the question:
    - Contains ANY educational terms (chapter, subject, teacher, section, class, topic, homework, exam)
    - Contains ANY performance-related terms (performance, progress, improvement, score, grade, learning)
    - Asks about ANY specific student, class, or academic entity
    - Requests ANY data about educational metrics or academic information
    - Contains follow-up questions related to previously mentioned academic entities
    - Uses pronouns (he/she/they/it) referring to previously discussed academic entities
    - Requests ANY comparative analysis between classes, students, or topics
    - Asks about teaching assignments or responsibilities
    - Contains implicit references to educational data (e.g., "Give me the summary" after discussing a class)
    - Asks about personal academic performance (e.g., "How am I doing?", "My recent performance")

    2. Tag as 'general_chat' ONLY when the question:
    - Is basic greeting or casual conversation (hi, hello, how are you)
    - Is about the AI itself or its capabilities
    - Is completely unrelated to education or academic contexts
    - Requests general knowledge not specific to educational data
    - Is small talk or pleasantries with no academic content

    Importantly:
    - When in doubt, classify as 'query_data'
    - If the question follows a 'query_data' conversation, maintain that classification unless clearly unrelated
    - ANY question that might require accessing student or class data should be 'query_data'

    Examples:
    Question: "Show me the performance of class 7B"
    query_data

    Question: "How is Shivansh doing?"
    query_data

    Question: "How are you today?"
    general_chat

    Classify this question: {question}

    Output only one of the two options (query_data or general_chat):"""

    prompt = PromptTemplate(
        template=template,
        input_variables=["question", "chat_history"]
    )
    
    def clean_response(response: str) -> str:
        """Clean the response to ensure it's either 'query_data' or 'general_chat'"""
        response = response.strip().lower()
        return "query_data" if "query_data" in response else "general_chat"
    
    # Return a callable function rather than a chain object
    def classify(inputs):
        filled_prompt = prompt.format(**inputs)
        response = llm.invoke(filled_prompt)
        return clean_response(response)
    
    return classify

def handle_query_data(question: str, db, streaming_handler, messages: list, tag: str, memory, process_collector=None) -> tuple:
    """Handle data queries with SQL chain with process tracking"""
    llm = Ollama(
        model="gemma3:12b",
        temperature=0,
        callbacks=[streaming_handler] if streaming_handler else None
    )
    
    # Format chat history properly for messages
    chat_history = format_chat_history(memory.chat_memory.messages if memory else messages)
    
    chain = create_chain(db, st.session_state.teacher_id, messages, memory, process_collector)
   
    results = chain.invoke({
        "question": question.lower(),
        "teacher_id": st.session_state.teacher_id,
        "chat_history": chat_history
    })
    
    st.session_state.current_df = results["dataframe"]
    
    # Get the answer prompt template
    _, answer_prompt = get_prompts()

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

    save_conversation_to_vector_store(
        question=question.lower(), 
        query=results["query"], 
        response=response,
        sql_result=results["dataframe"],
        tag=tag
    )

    return response, results


def follow_up_question(
    question: str, 
    streaming_handler, 
    messages: list, 
    response: str, 
    results: dict,
    memory,
    process_collector=None
) -> str:
    llm = Ollama(
        model="gemma3:12b",
        temperature=0.3,
        callbacks=[streaming_handler] if streaming_handler else None
    )

    chat_context = format_chat_history(messages)
    
    # Get database schema information
    db = init_db()
    table_info = db.get_table_info()

    follow_up_prompt = PromptTemplate(
        input_variables=["input", "response", "result", "chat_history", "table_info"],
        template="""
# Follow-up Question Generator

Generate EXACTLY 3 simple and generaic follow-up questions in natural teacher language.

## Database Schema
{table_info}

## Context
Original Question: {input}
Response: {response}
Result Data: {result}
Previous Conversation: {chat_history}

## CRITICAL RULES:
- Generate 3 simple follow up questions for current question to create sql queries as per Database schema and Result data.
- Generate follow up questions in SIMPLE ENGLISH exactly as a teacher would naturally ask.
- Never ask Original Question again in follow-up questions.
- Create follow up questions that can create SQL queries based on the SQL result data.
- Create follow up questions based on the SQL result data only and previous conversations
- Create follow up questions start with "What", "How many", "Which", "How", "Can you tell me", "Show me" etc only as asking data.
- Create student based follow up questions like "Which student.." or "How many.." .
- When SQL Results are empty or None, Repeat the previous questions based on conversation history with SQL result not None or Empty, Do not create new questions.
- Questions should be short, clear, and conversational
- Do not generate questions with subjects like "math", "science", etc.
- Use everyday classroom language (no technical terms like CR, LI, etc.)
- Questions must be answerable using the database schema provided
- Each question should ask about something different
- STICK TO entities in the results (students, classes, chapters mentioned)
- DO NOT create new names or data not in the results
- Phrase questions as a teacher speaking to an assistant

## EXAMPLES OF GOOD TEACHER QUESTIONS:
- "How are my students doing in class 8d?"
- "Show me which students need help with friction."
- "What's the progress in science class?"
- "Which chapters are my students struggling with?"

## BAD EXAMPLES (DO NOT USE):
- "What is the completion rate in section 7a?" (too technical)
- "Show learning index for student Priya." (uses technical term)
- "List all students with LG count below threshold." (too technical)
- "Can you tell me which class is having the most trouble with the material?" (No sql can be generated)
- "how is rahul performing compared to last month?" (No rahul in the results or name is incomplete)

Follow-Up Questions:
1. 
2. 
3. 

"""
    )

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

    follow_up_response = llm.invoke(follow_up_prompt.format(**follow_up_input))
    
    if process_collector:
        process_collector.add_step("8: the follow up questions", follow_up_response)

    save_conversation_to_vector_store(
        question=question,
        response=f"Follow-Up Questions:\n{follow_up_response}",
        tag="follow_up_questions"
    )

    return follow_up_response

# def follow_up_question(
#     question: str, 
#     streaming_handler, 
#     messages: list, 
#     response: str, 
#     results: pd.DataFrame,
#     memory
# ) -> str:
#     llm = Ollama(
#         model="gemma3:12b",
#         temperature=0.3,
#         callbacks=[streaming_handler] if streaming_handler else None
#     )

#     chat_context = format_chat_history(messages)

#     follow_up_prompt = PromptTemplate(
#         input_variables=["input", "response", "result", "chat_history"],
#         template="""
# # Follow-up Question Generator

# You are generating 3 follow-up questions for teachers based on the original question, response data, and conversation history.

# ## Data Context
# Original Question: {input}
# Response: {response}
# Result Data: {result}
# Previous Conversation: {chat_history}

# ## Question Guidelines

# 1. Generate exactly 3 different questions that:
#    - Focus on student progress, chapter performance, or class achievement
#    - Use simple language (avoid technical terms like "completion rate" or "learning index")
#    - Relate directly to data mentioned in the response
#    - Build upon, but don't repeat, the original question
#    - Include specific student/class/chapter names from the results
#    - Are phrased as data retrieval questions (using "what", "how many", "which", etc.)

# 2. Structure your questions to cover different perspectives:
#    - Question 1: Focus on student-level insights
#    - Question 2: Focus on class/section-level insights
#    - Question 3: Focus on chapter-level insights (change context from original question)

# 3. Translation guide (use these in questions):
#    - Use "performance" instead of "learning index"
#    - Use "progress" instead of "completion rate"
#    - Use "chapters" and "topics" instead of "learning units"
#    - Use "attempted" instead of "engaged"

# ## Requirements
# - No questions about teacher IDs
# - No questions about learning gaps
# - No repetition of previous questions
# - No engagement level questions
# - No technical terms (LI, CR, etc.)
# - If result data is empty, generate new contextual questions based on conversation history
# - Ensure all questions relate to academic performance metrics

# Follow-Up Questions:
# 1. 
# 2. 
# 3. 
# """
#     )

#     # Prepare the input
#     follow_up_input = {
#         "input": question,
#         "response": response,
#         "result": results,
#         "chat_history": chat_context
#     }

#     follow_up_response = llm.invoke(follow_up_prompt.format(**follow_up_input))

#     # Extract just the numbered questions if the model includes extra text
#     # import re
#     # questions_pattern = r'(?:Follow-Up Questions:)?\s*(?:\d\.\s*(.*?)(?=\d\.\s*|\Z)){3}'
#     # matches = re.findall(r'\d\.\s*(.*?)(?=\d\.\s*|\Z|\n\n)', follow_up_response, re.DOTALL)
    
#     # if len(matches) >= 3:
#     #     formatted_response = "Follow-Up Questions:\n1. " + matches[0].strip() + "\n2. " + matches[1].strip() + "\n3. " + matches[2].strip()
#     # else:
#     #     formatted_response = follow_up_response

#     save_conversation_to_vector_store(
#         question=question,
#         response=f"Follow-Up Questions:\n{follow_up_response}",
#         tag="follow_up_questions"
#     )

#     return follow_up_response



def handle_general_chat(question: str, streaming_handler, messages: list, tag: str, process_collector=None) -> str:
    llm = Ollama(
        model="gemma3:12b",
        temperature=0.3,
        callbacks=[streaming_handler] if streaming_handler else None
    )

    chat_context = format_chat_history(messages)

    prompt = PromptTemplate.from_template(
        """You are a friendly and helpful AI Assistant .You will answering Teachers questions as Answer accordingly. Respond naturally while following these guidelines:


        Previous conversation:
        {chat_history}

        CRITICAL:
        - Always  consider the previous conversation context and the question asked.
        - Use the previous conversation context to understand the question and provide a relevant answer.
        - Do not generate questions if previous question has no data in the response and result data.
        - Do not answer data related question with any related data . Questions contains word like (chapter , section , performance , improvement , topics , class, subject etc)
        - Answer basic questions about any topic
        - Provide information about general knowledge topics
        - Assist with basic problem-solving
        - Engage in friendly conversation
        - Give simple explanations for various concepts
        - Help with basic planning and organization

        Response Guidelines:
        - Do not generate answer for Data related question .Questions contains word like (chapter , section , performance , improvement , topics , class, subject etc)
        - Be friendly and conversational
        - Use simple, clear language
        - Give practical examples when needed
        - Be encouraging and supportive
        - Keep explanations concise but informative
        - Relate to real-life situations when possible
        - Start responses directly and naturally
        - End with complete thoughts
        - Maintain a helpful and positive tone

        Question: {question}
        Response:
        """
    )

    response = llm.invoke(prompt.format(chat_history=chat_context, 
                                        question=question))
    
    if process_collector:
        process_collector.add_step("4: the general answer", response)
    
    save_conversation_to_vector_store(question, response=response, tag=tag)
    
    return response


def process_question(question: str) -> Dict:
    try:
        llm = Ollama(model="gemma3:12b", temperature=0.1)
        tagger_chain = create_tagger(llm)
        
        tag_result = tagger_chain.invoke({"question": question})
        
        if tag_result.tool == "query_data":
            db = SQLDatabase.from_uri("sqlite:///acadally.db")
            result = handle_query_data(question, db)
        else:
            result = handle_general_chat(question)
        
        return result
        
    except Exception as e:
        return {
            "response": f"An error occurred: {str(e)}",
            "dataframe": None
        }

def main():
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

    # Sidebar settings
    with st.sidebar:
        st.header("Settings")
        new_teacher_id = st.text_input("Teacher ID", value=st.session_state.teacher_id)
        
        if new_teacher_id != st.session_state.teacher_id:
            st.session_state.teacher_id = new_teacher_id
            st.session_state.current_df = None
            memory.clear()
            clear_chat_history(st.session_state.teacher_id)
            st.rerun()

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
                    # --- PRESERVING ORIGINAL STREAMING CODE ---
                    save_conversation_to_vector_store(question.lower())
                    streaming_handler = StreamlitCallbackHandler(chat_container, message_placeholder)
                    llm = Ollama(model="gemma3:12b", temperature=0.2)
                    
                    # Format chat history properly
                    chat_history = format_chat_history(memory.chat_memory.messages)
                    
                    # CRITICAL FIX: Call the tagger function directly, not with invoke
                    tagger_func = create_tagger(llm)
                    # Notice we're calling the function directly, not using .invoke()
                    tool_type = tagger_func({
                        "question": question.lower(),
                        "chat_history": chat_history
                    })
                    process.add_step("3: tagger", tool_type)

                    if tool_type == "query_data":
                        # Process query data with chat history
                        db = init_db()
                        response, results = handle_query_data(
                            question, 
                            db, 
                            streaming_handler, 
                            memory.chat_memory.messages,  
                            tag=tool_type, 
                            memory=memory,
                            process_collector=process  # Pass process collector
                        )

                        # Generate follow-up questions with chat history
                        follow_up_questions = follow_up_question(
                            question=question.lower(),
                            streaming_handler=streaming_handler,
                            response=response,
                            results=results,
                            messages=memory.chat_memory.messages,
                            memory=memory,
                            process_collector=process  # Pass process collector
                        )

                        # Combine main response and follow-up questions into a single response
                        full_response = f"{response}\n\nFollow-Up Questions:\n{follow_up_questions}"

                        # Handle DataFrame display
                        df = results.get("dataframe", pd.DataFrame())
                        if isinstance(df, pd.DataFrame) and not df.empty:
                            with df_placeholder:
                                st.dataframe(df, use_container_width=True)

                            # Add response with DataFrame to memory
                            memory.chat_memory.add_message(
                                AIMessage(
                                    content=full_response,
                                    additional_kwargs={"dataframe": df}
                                )
                            )
                        else:
                            memory.chat_memory.add_message(AIMessage(content=full_response))
                    else:
                        # Handle general chat with chat history
                        response = handle_general_chat(
                            question.lower(), 
                            streaming_handler, 
                            memory.chat_memory.messages, 
                            tag=tool_type,
                            process_collector=process  # Pass process collector
                        )
                        memory.chat_memory.add_message(AIMessage(content=response))
                    
                    # Store process steps in session state
                    st.session_state.process_steps[question.lower()] = process.get_steps()

                    st.rerun()

            except Exception as e:
                error_message = f"An error occurred: {str(e)}"
                st.error(error_message)
                process.add_step("Error", error_message)
                st.session_state.process_steps[question.lower()] = process.get_steps()
                save_conversation_to_vector_store(
                    question=question.lower(),
                    response=error_message
                )
                memory.chat_memory.add_message(AIMessage(content=error_message))

    if st.sidebar.button("Clear Chat History"):
        memory.clear()
        st.session_state.current_df = None
        clear_chat_history(st.session_state.teacher_id)
        st.rerun()
if __name__ == "__main__":
    main()

