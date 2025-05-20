# Text-to-SQL AI Assistant

![Python Version](https://img.shields.io/badge/python-3.9%2B-brightgreen)
![Streamlit](https://img.shields.io/badge/streamlit-1.31.1-red)
![LangChain](https://img.shields.io/badge/langchain-0.0.353-orange)

A powerful AI assistant that translates natural language questions into SQL queries. Ask questions about your data in plain English and get instant insights through SQL query generation and visualization. While the example implementation uses an educational database, the architecture can be adapted to any SQL database.

![Application Screenshot](https://raw.githubusercontent.com/username/text-to-sql-assistant/main/docs/screenshot.png)

## 🌟 Features

- **Natural Language to SQL**: Convert everyday questions into precise SQL queries
- **SQL Query Generation**: Automatically translates questions into optimized database queries
- **Data Visualization**: Displays query results with interactive tables
- **Contextual Memory**: Maintains conversation context for follow-up questions
- **Follow-up Suggestions**: Suggests relevant follow-up questions based on query results
- **Vector Storage**: Learns from previous questions to improve response accuracy
- **Process Tracking**: Detailed process steps for debugging and transparency
- **Modular Architecture**: Easily adaptable to different databases and use cases

## 🛠️ Technologies

- **Frontend**: Streamlit
- **NLP**: LangChain & Ollama (Gemma 3 12B model)
- **Database**: SQLite (easily configurable for other SQL databases)
- **Embeddings**: SentenceTransformerss
- **Vector Store**: ChromaDB

## 📋 Prerequisites

- Python 3.9+
- Ollama installed and running
- SQLite database (or any other SQL database)
- 8GB+ RAM recommended

## 🚀 Installation

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/text-to-sql-assistant.git
cd text-to-sql-assistant
```

2. **Create a virtual environment**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Install and start Ollama**

```bash
# Install Ollama following instructions at https://ollama.com/download
ollama pull gemma3:12b
ollama serve
```

5. **Configure your database**

Edit `config/settings.py` to set your database connection string:

```python
# Database settings
DB_URI = "sqlite:///your_database.db"  # Replace with your database URI
```

6. **Run the application**

```bash
streamlit run app.py
```

## 💻 Usage

1. Enter your user ID in the sidebar (default: 8933)
2. Type your question in the chat input
3. View the AI's response, including generated SQL, data tables, and insights
4. Ask follow-up questions or select from suggested questions
5. Clear chat history using the sidebar button when needed

## 📊 Example Queries

Using the included example educational database:

- "Show me the overall performance summary of all classes"
- "Which students need help with the friction chapter?"
- "What's the average time spent on each chapter?"
- "Which chapters have the best performance across all classes?"
- "Top performing students in class 6D"
- "How many learning units have been engaged across all sections?"

You can adapt these to your own database schema by modifying the examples in `ai/prompts.py`.

## 🔄 Adapting to Your Database

To use this application with your own database:

1. Update the database connection string in `config/settings.py`
2. Modify the example queries in `ai/prompts.py` to match your schema
3. Update the schema information in the SQL prompt template
4. Adjust the formatting in the answer prompt to match your data model

## 🏗️ Project Structure

```
text_to_sql_assistant/
├── app.py                     # Main application entry point
├── config/                    # Configuration settings
│   ├── __init__.py
│   └── settings.py
├── database/                  # Database operations
│   ├── __init__.py
│   ├── connection.py
│   └── queries.py
├── ai/                        # AI components
│   ├── __init__.py
│   ├── llm.py
│   ├── chains.py
│   ├── prompts.py
│   ├── tagger.py
│   └── handlers.py
├── memory/                    # Memory management
│   ├── __init__.py
│   ├── conversation.py
│   └── vector_store.py
├── ui/                        # UI components
│   ├── __init__.py
│   ├── main_view.py
│   └── sidebar.py
└── utils/                     # Utilities
    ├── __init__.py
    ├── formatters.py
    ├── logging.py
    ├── error_handling.py
    └── session.py
```

## 🙏 Acknowledgements

- [LangChain](https://github.com/langchain-ai/langchain) for the language model framework
- [Streamlit](https://streamlit.io/) for the web application framework
- [Ollama](https://ollama.com/) for the local LLM implementation
- [ChromaDB](https://github.com/chroma-core/chroma) for vector storage

---

<p align="center">Convert natural language to SQL with ease</p>.py
│   ├── prompts.py
│   ├── tagger.py
│   └── handlers.py
├── memory/                    # Memory management
│   ├── __init__.py
│   ├── conversation.py
│   └── vector_store.py
├── ui/                        # UI components
│   ├── __init__.py
│   ├── main_view.py
│   └── sidebar.py
└── utils/                     # Utilities
    ├── __init__.py
    ├── formatters.py
    ├── logging.py
    ├── error_handling.py
    └── session.py
```

## 🙏 Acknowledgements

- [LangChain](https://github.com/langchain-ai/langchain) for the language model framework
- [Streamlit](https://streamlit.io/) for the web application framework
- [Ollama](https://ollama.com/) for the local LLM implementation
- [ChromaDB](https://github.com/chroma-core/chroma) for vector storage

---

<p align="center">Made with ❤️ for teachers and students</p>
