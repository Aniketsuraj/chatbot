from langchain_core.prompts import PromptTemplate
from ai.llm import get_llm

def create_tagger(llm=None):
    """
    Create a function for tagging questions as either 'query_data' or 'general_chat'.
    
    Args:
        llm: LLM instance to use for classification. If None, a new one will be created.
        
    Returns:
        A callable function that accepts a dict with 'question' and 'chat_history' keys
        and returns the classification tag.
    """
    if llm is None:
        llm = get_llm(temperature=0.1)
        
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