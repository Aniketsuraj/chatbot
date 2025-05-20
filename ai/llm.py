from langchain_community.llms import Ollama
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from config.settings import LLM_MODEL, DEFAULT_TEMPERATURE, QUERY_TEMPERATURE, CHAT_TEMPERATURE

class StreamlitCallbackHandler(StreamingStdOutCallbackHandler):
    """Callback handler for streaming LLM output to a Streamlit container."""
    
    def __init__(self, container, text_placeholder):
        super().__init__()
        self.container = container
        self.text_placeholder = text_placeholder
        self.text = ""

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.text_placeholder.markdown(self.text)

def get_llm(temperature=None, streaming_handler=None, model=None):
    """Get an LLM instance with the specified parameters."""
    if temperature is None:
        temperature = DEFAULT_TEMPERATURE
    
    if model is None:
        model = LLM_MODEL
    
    # Handle callbacks properly - note that Ollama doesn't use streaming param
    if streaming_handler:
        return Ollama(
            model=model,
            temperature=temperature,
            callbacks=[streaming_handler]
        )
    else:
        return Ollama(
            model=model,
            temperature=temperature
        )

def get_query_llm(streaming_handler=None):
    """Get an LLM instance optimized for query generation."""
    return get_llm(temperature=QUERY_TEMPERATURE, streaming_handler=streaming_handler)

def get_chat_llm(streaming_handler=None):
    """Get an LLM instance optimized for conversational responses."""
    return get_llm(temperature=CHAT_TEMPERATURE, streaming_handler=streaming_handler)