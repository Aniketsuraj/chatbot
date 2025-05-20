import streamlit as st
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate

@st.cache_data
def get_prompts():
    """Get cached prompt templates for SQL query generation and answer formatting."""
    # Example prompt for few-shot learning
    example_prompt = PromptTemplate(
        input_variables=["input", "query"],
        template="Input: {input}\nSQL Query: {query}"
    )

    # Examples for few-shot learning
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

    # SQL query generation prompt
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

    # Answer formatting prompt
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
Answer: [Direct answer to the question with specific insights from the data]
Recommendation : [1-2 specific content areas or skills revealed by the data that need attention]
        """
    )

    # Follow-up questions prompt
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

    # General chat prompt
    general_chat_prompt = PromptTemplate.from_template(
        """You are a friendly and helpful AI Assistant. You will answering Teachers questions as Answer accordingly. Respond naturally while following these guidelines:

        Previous conversation:
        {chat_history}

        CRITICAL:
        - Always consider the previous conversation context and the question asked.
        - Use the previous conversation context to understand the question and provide a relevant answer.
        - Do not generate questions if previous question has no data in the response and result data.
        - Do not answer data related question with any related data. Questions contains word like (chapter, section, performance, improvement, topics, class, subject etc)
        - Answer basic questions about any topic
        - Provide information about general knowledge topics
        - Assist with basic problem-solving
        - Engage in friendly conversation
        - Give simple explanations for various concepts
        - Help with basic planning and organization

        Response Guidelines:
        - Do not generate answer for Data related question. Questions contains word like (chapter, section, performance, improvement, topics, class, subject etc)
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

    return sql_prompt, answer_prompt, follow_up_prompt, general_chat_prompt