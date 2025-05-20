import pandas as pd
import re

class DataFrameSQLTool:
    """Tool for executing SQL queries against a database and returning results as a DataFrame."""
    
    def __init__(self, db):
        """Initialize with a SQLDatabase instance."""
        self.db = db
    
    def clean_query(self, query: str) -> str:
        """Clean and validate SQL query."""
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
        """Execute SQL query and return results."""
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

def clean_query(query: str) -> str:
    """Clean an SQL query from extra content."""
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