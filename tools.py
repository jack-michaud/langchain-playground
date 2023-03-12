import subprocess
from milvus_db import vector_db_wrapper, add_new_document_to_vector_db
from langchain.docstore.document import Document

from langchain.agents import tool


@tool("ask_for_approval")
def ask_for_approval(query: str) -> str:
    """Asks Jack for feedback on a response. Rephrase the response if a question is required."""
    # Uses zenity to ask for feedback
    response = subprocess.check_output(["zenity", "--entry", "--text", query])

    return f'Jack said: {response.decode("utf-8").strip()}'


@tool("send_message")
def send_message(message: str) -> str:
    """Send a message after approval. All messages must be approved first."""
    message = message.strip()
    subprocess.check_output(["notify-send", "Copy to clipboard", message])
    subprocess.check_output(["wl-copy", message])
    return f'Message sent: "{message}"'


@tool("query_pkm")
def query_personal_knowledge_management(query: str) -> str:
    """Query Jack's personal wiki."""
    # Uses rofi to ask for feedback
    response = subprocess.check_output(["rofi", "-dmenu", "-p", query])

    return f'You said: {response.decode("utf-8").strip()}'

@tool("add_memory")
def add_memory(document: str) -> str:
    """Store a memory."""
    milvus = vector_db_wrapper()
    doc = Document(page_content=document)
    add_new_document_to_vector_db(milvus, doc)
    return f'I will contemplate what was said here.'

@tool("search_memory")
def search_memory(query: str) -> str:
    """Store a memory."""
    milvus = vector_db_wrapper()
    return milvus.query(query)
